from flask import Flask, request, jsonify
import sqlite3
import os
import requests
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import sqlparse  # For parsing SQL queries
import re  # For schema parsing
import traceback  # For detailed error logging
import json

# Initialize Flask app
app = Flask(__name__)

# database
DATABASE_FILE_PATH = os.path.join("formula_1", "formula_1.sqlite")

# Hugging Face Configuration
HF_API_URL = os.environ.get("HF_API_URL")
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_AUTH_TOKEN = f"Bearer {HF_TOKEN}" if HF_TOKEN and not HF_TOKEN.startswith("Bearer ") else HF_TOKEN

# OpenAI Configuration
OPENAI_API_KEY_FROM_ENV = os.environ.get("OPENAI_API_KEY")
FALLBACK_OPENAI_API_KEY = " "  # Your specific key

if OPENAI_API_KEY_FROM_ENV and OPENAI_API_KEY_FROM_ENV != "your-openai-api-key":
    OPENAI_API_KEY = OPENAI_API_KEY_FROM_ENV
    print("INFO: Using OpenAI API Key from environment variable.")
else:
    OPENAI_API_KEY = FALLBACK_OPENAI_API_KEY
    print("INFO: Using fallback hardcoded OpenAI API Key. Ensure this is intended for local testing only.")

if not OPENAI_API_KEY or "your-openai-api-key" in OPENAI_API_KEY:  # Final check
    raise ValueError(
        "CRITICAL: Valid OpenAI API Key is not available. Please set OPENAI_API_KEY environment variable or check hardcoded value.")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY  # Make it available for Langchain
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2,
                 max_tokens=300)  # Increased max_tokens for potentially longer SQL

# Gemini API Configuration
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", " ")
if not GEMINI_API_KEY or GEMINI_API_KEY == "your-gemini-api-key":
    raise ValueError("CRITICAL: Valid Gemini API Key is not available. Please set GEMINI_API_KEY environment variable.")
import google.generativeai as genai
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

optimization_prompt_template = """
### Instruction:
You are an expert SQL optimizer and corrector.
The user wants to query a database with the following schema.
The user asked the following question.
An initial attempt to generate SQL produced the 'Raw SQL Query' below.

Your goal is to produce a **single, correct, and executable SQL query against the provided schema** that accurately answers the question.
{error_feedback}
- Ensure all column and table names exactly match the provided schema (case-sensitive if the database is).
- Pay close attention to how specific concepts like 'wins' or 'fastest laps' are represented in the schema (e.g., a 'win' is often by checking `results.position = '1'` or `results.positionOrder = 1`; a 'fastest lap' might be in a 'lapTimes' table by looking for minimum lap time in milliseconds, or in a 'results' table via columns like `fastestLapTime` or `rank = '1'`).
- If the 'Raw SQL Query' is nonsensical or completely unrelated, please generate a new SQL query based *only* on the Schema and Question.
- Return only the single, optimized, and corrected SQL query. Do not include any explanations or surrounding text.

### Schema:
{schema}

### Question:
{question}

### Raw SQL Query:
{raw_sql}

### Optimized SQL:
"""
optimization_prompt = PromptTemplate(
    input_variables=["schema", "question", "raw_sql"],
    partial_variables={"error_feedback": ""},
    template=optimization_prompt_template
)
optimization_chain = optimization_prompt | llm | StrOutputParser()


# --- Helper Functions ---
# to get database schema
def get_db_schema(db_path):
    agent_log = []
    schema_str = ""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        agent_log.append(f"Found tables: {[table[0] for table in tables]}")

        if not tables:
            error_msg = "No tables found in the database."
            agent_log.append(error_msg)
            conn.close()
            return None, agent_log

        for table_name_tuple in tables:
            table_name = table_name_tuple[0]
            cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name = ?", (table_name,))
            create_statement_tuple = cursor.fetchone()
            if create_statement_tuple and create_statement_tuple[0]:
                schema_str += create_statement_tuple[0] + ";\n\n"
            else:
                agent_log.append(
                    f"Warning: No CREATE statement for table {table_name}. Constructing manually using PRAGMA.")
                cursor.execute(f"PRAGMA table_info('{table_name}');")
                columns_info = cursor.fetchall()
                if columns_info:
                    column_defs = []
                    for col_info in columns_info:
                        col_name, col_type, notnull, dflt_value, pk = col_info[1], col_info[2], col_info[3], col_info[
                            4], col_info[5]
                        col_type = col_type if col_type else "TEXT"
                        col_def = f'"{col_name}" {col_type}'
                        if notnull: col_def += " NOT NULL"
                        if dflt_value is not None:
                            if isinstance(dflt_value, str) and dflt_value.upper() not in ['CURRENT_TIMESTAMP',
                                                                                          'CURRENT_DATE',
                                                                                          'CURRENT_TIME',
                                                                                          'NULL'] and not dflt_value.lstrip(
                                    '-').isnumeric():
                                escaped_dflt_value = dflt_value.replace("'", "''")
                                col_def += f" DEFAULT '{escaped_dflt_value}'"
                            else:
                                col_def += f" DEFAULT {dflt_value}"
                        if pk: col_def += " PRIMARY KEY"
                        column_defs.append(col_def)
                    if column_defs:
                        schema_str += f"CREATE TABLE \"{table_name}\" ({', '.join(column_defs)});\n\n"
                else:
                    agent_log.append(f"Warning: No columns found via PRAGMA for table {table_name}.")

        conn.close()
        agent_log.append("Successfully extracted schema.")
    except sqlite3.Error as e:
        error_msg = f"SQLite error while getting schema: {e}"
        print(error_msg)
        agent_log.append(error_msg)
        return None, agent_log
    except Exception as e:
        error_msg = f"Unexpected error while getting schema: {e}"
        print(error_msg)
        traceback.print_exc()  # Print full traceback for unexpected errors
        agent_log.append(error_msg)
        return None, agent_log
    return schema_str.strip(), agent_log


def parse_schema_to_dict(schema_str: str, agent_log: list):
    parsed_schema = {}
    create_table_pattern = re.compile(
        r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(?:\"([\w\s]+)\"|`([\w\s]+)`|([\w]+))\s*\((.*?)\)\s*;",
        re.IGNORECASE | re.DOTALL
    )
    column_pattern = re.compile(r"^\s*(?:\"([\w\s]+)\"|`([\w\s]+)`|([\w]+))\s+.*", re.IGNORECASE)

    for table_match in create_table_pattern.finditer(schema_str):
        table_name = (table_match.group(1) or table_match.group(2) or table_match.group(3))
        if not table_name: continue
        table_name_lower = table_name.lower()

        columns_block = table_match.group(4)
        current_columns = set()

        col_defs_raw = []
        paren_level = 0
        current_def_item = ""
        for char_item in columns_block:
            if char_item == '(':
                paren_level += 1
            elif char_item == ')':
                paren_level -= 1
            elif char_item == ',' and paren_level == 0:
                col_defs_raw.append(current_def_item.strip())
                current_def_item = ""
                continue
            current_def_item += char_item
        if current_def_item.strip(): col_defs_raw.append(current_def_item.strip())

        for col_def_str_item in col_defs_raw:
            col_name_match = column_pattern.match(col_def_str_item)
            if col_name_match:
                col_name = (col_name_match.group(1) or col_name_match.group(2) or col_name_match.group(3))
                if col_name:
                    current_columns.add(col_name.lower())

        if current_columns:
            parsed_schema[table_name_lower] = current_columns

    agent_log.append(
        f"Parsed schema dictionary for validation: {parsed_schema if parsed_schema else 'No tables parsed'}")
    return parsed_schema


def validate_sql_against_schema(sql_query: str, parsed_schema_dict: dict, agent_log: list):
    validation_errors = []
    if not parsed_schema_dict:
        agent_log.append("Validation skipped: Parsed schema dictionary is empty.")
        return []
    try:
        parsed = sqlparse.parse(sql_query)
        if not parsed:
            validation_errors.append("SQL query is empty or could not be parsed by sqlparse.")
            return validation_errors
        stmt = parsed[0]

        def extract_identifiers_from_tokenlist(tokenlist):
            tables, columns = set(), set()
            after_clause_keyword = False
            clause_keywords = ("FROM", "JOIN", "LEFT JOIN", "RIGHT JOIN", "INNER JOIN", "UPDATE", "INTO", "TABLE")

            for token in tokenlist.tokens:
                if token.is_whitespace or token.ttype in (sqlparse.tokens.Comment.Single,
                                                          sqlparse.tokens.Comment.Multiline):
                    continue

                if token.is_keyword and token.normalized in clause_keywords:
                    after_clause_keyword = True
                    continue

                if isinstance(token, sqlparse.sql.Identifier):
                    name = token.get_real_name().lower()
                    parent_name = token.get_parent_name().lower() if token.get_parent_name() else None
                    if parent_name:
                        columns.add((name, parent_name))
                    elif after_clause_keyword:
                        tables.add(name)
                    else:
                        columns.add((name, None))  # Unqualified column
                    after_clause_keyword = False  # Reset after consuming identifier
                elif isinstance(token, sqlparse.sql.IdentifierList):
                    for identifier in token.get_identifiers():
                        if isinstance(identifier, sqlparse.sql.Identifier):
                            name = identifier.get_real_name().lower()
                            parent_name = identifier.get_parent_name().lower() if identifier.get_parent_name() else None
                            if parent_name:
                                columns.add((name, parent_name))
                            else:
                                columns.add((name, None))
                        elif identifier.is_group:
                            sub_t, sub_c = extract_identifiers_from_tokenlist(identifier)
                            tables.update(sub_t);
                            columns.update(sub_c)
                    after_clause_keyword = False
                elif token.is_group:
                    sub_t, sub_c = extract_identifiers_from_tokenlist(token)
                    tables.update(sub_t);
                    columns.update(sub_c)
                elif token.is_keyword:  # Any other keyword resets context
                    after_clause_keyword = False
            return tables, columns

        query_tables, query_columns_with_qualifiers = extract_identifiers_from_tokenlist(stmt)
        agent_log.append(
            f"Validation: Identified tables: {query_tables}, Columns (col,qualifier): {query_columns_with_qualifiers}")

        for table_name in query_tables:
            if table_name not in parsed_schema_dict:
                validation_errors.append(
                    f"Table '{table_name}' not found in schema. Known: {list(parsed_schema_dict.keys())}")

        for col_name, qualifier in query_columns_with_qualifiers:
            if col_name == '*': continue
            if qualifier:
                if qualifier not in parsed_schema_dict:
                    found_col_for_alias = False
                    for schema_table in query_tables:  # Check if col_name exists in any of the tables actually in the query
                        if schema_table in parsed_schema_dict and col_name in parsed_schema_dict[schema_table]:
                            found_col_for_alias = True;
                            break
                    if not found_col_for_alias:
                        validation_errors.append(
                            f"Column '{qualifier}.{col_name}' - Table/Alias '{qualifier}' not in schema, and column not found in other queried tables.")
                elif col_name not in parsed_schema_dict[qualifier]:
                    validation_errors.append(
                        f"Column '{col_name}' not found in table '{qualifier}'. Available: {sorted(list(parsed_schema_dict[qualifier]))}.")
            else:  # Unqualified column
                found_unqualified = False
                if len(query_tables) == 1:
                    table_in_query = list(query_tables)[0]
                    if table_in_query in parsed_schema_dict and col_name in parsed_schema_dict[table_in_query]:
                        found_unqualified = True
                    elif table_in_query not in parsed_schema_dict:
                        validation_errors.append(
                            f"Table '{table_in_query}' for unqualified column '{col_name}' not found in schema.")
                    else:  # Table is known, but column is not in it
                        validation_errors.append(
                            f"Unqualified column '{col_name}' not found in the single query table '{table_in_query}'. Available: {sorted(list(parsed_schema_dict[table_in_query]))}.")
                else:  # Multiple tables, check all
                    for t_name in query_tables:
                        if t_name in parsed_schema_dict and col_name in parsed_schema_dict[t_name]:
                            found_unqualified = True;
                            break
                    if not found_unqualified:
                        validation_errors.append(
                            f"Unqualified column '{col_name}' not found in any of the queried schema tables: {list(query_tables)}.")

        if validation_errors:
            agent_log.append(f"Validation: Errors: {validation_errors}")
        else:
            agent_log.append("Validation: SQL consistent with schema.")
    except Exception as e:
        error_msg = f"Error during SQL validation: {str(e)}"
        print(error_msg);
        agent_log.append(error_msg);
        validation_errors.append(error_msg)
    return validation_errors


def query_hf_endpoint(schema_text, question_text, agent_log):
    if not HF_API_URL or not HF_AUTH_TOKEN:

        error_msg = "Error: Hugging Face API URL or Token is not configured."
        print(error_msg);
        agent_log.append(error_msg)
        return None, agent_log, error_msg
    if not schema_text:

        error_msg = "Error: Schema is empty. Cannot query Hugging Face without a schema."
        print(error_msg);
        agent_log.append(error_msg)
        return None, agent_log, error_msg

    headers = {"Authorization": HF_AUTH_TOKEN, "Content-Type": "application/json"}
    payload = {"inputs": {"schema": schema_text, "question": question_text}}
    agent_log.append(f"Sending payload to Hugging Face endpoint: {HF_API_URL}")
    try:
        response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        agent_log.append(f"Received response from HF: {result}")
        raw_sql = result.get("sql_query")
        if not raw_sql:

            error_msg = "Hugging Face did not return 'sql_query' or it was empty in its response."
            agent_log.append(error_msg);
            agent_log.append(f"Full HF response: {result}")
            return None, agent_log, error_msg
        return raw_sql, agent_log, None
    except requests.exceptions.RequestException as e:

        error_msg = f"Error querying Hugging Face endpoint: {e}"
        print(error_msg);
        agent_log.append(error_msg)

        if e.response is not None:
            try:
                error_detail = e.response.json(); agent_log.append(
                    f"HF Endpoint Error Detail: {error_detail}"); error_msg += f" Details: {error_detail.get('error', e.response.text)}"
            except ValueError:
                agent_log.append(
                    f"HF Endpoint Error (non-JSON): {e.response.text}"); error_msg += f" Details: {e.response.text}"
        return None, agent_log, error_msg
    except Exception as e:
        error_msg = f"Unexpected error during LLM query: {str(e)}"
        print(error_msg);
        agent_log.append(error_msg);
        traceback.print_exc()
        return None, agent_log, error_msg


def optimize_query_with_feedback(schema, question, raw_sql, agent_log, error_feedback_str=""):
    agent_log.append(
        f"Optimizing query with OpenAI. Feedback: '{error_feedback_str if error_feedback_str else 'None'}'")
    current_raw_sql_input = raw_sql if isinstance(raw_sql,
                                                  str) and raw_sql.strip() else "/* No valid previous SQL attempt or empty, please generate based on schema and question. */"
    if not raw_sql and not error_feedback_str:  # If really no prior SQL and no feedback
        current_raw_sql_input = "/* Please generate a new SQL query based on the schema and question. */"

    try:
        current_prompt_with_feedback = optimization_prompt.partial(error_feedback=error_feedback_str)
        chain_with_feedback = current_prompt_with_feedback | llm | StrOutputParser()
        optimized_sql = chain_with_feedback.invoke({
            "schema": schema, "question": question, "raw_sql": current_raw_sql_input
        })
        agent_log.append(f"Optimized SQL from OpenAI: {optimized_sql}")
        if not optimized_sql or not any(keyword in optimized_sql.upper() for keyword in
                                        ["SELECT", "INSERT", "UPDATE", "DELETE", "CREATE", "WITH", "DROP"]):

            agent_log.append(f"Optimization failed to produce valid SQL. Raw optimized response: {optimized_sql}")
            return None, agent_log, "Optimization did not generate recognized SQL structure"
        return optimized_sql, agent_log, None
    except Exception as e:

        error_msg = f"Error during query optimization: {str(e)}"
        agent_log.append(error_msg);
        print(error_msg);
        traceback.print_exc()
        return None, agent_log, error_msg


def execute_sql_query(db_path, sql_query):

    log_updates = [];
    conn = None
    try:
        conn = sqlite3.connect(db_path);
        cursor = conn.cursor()
        log_updates.append(f"Executing SQL: {sql_query}");
        cursor.execute(sql_query)
        results = cursor.fetchall()
        column_names = [desc[0] for desc in cursor.description] if cursor.description else []
        data = [dict(zip(column_names, row)) for row in results]
        log_updates.append(f"SQL execution successful. Fetched {len(data)} rows.")
        return data, column_names, None, log_updates
    except sqlite3.Error as e:

        error_msg = f"SQLite error: {e}";
        print(error_msg);
        log_updates.append(error_msg)
        return None, None, error_msg, log_updates
    except Exception as e:

        error_msg = f"Execution error: {e}";
        print(error_msg);
        log_updates.append(error_msg);
        traceback.print_exc()
        return None, None, error_msg, log_updates
    finally:
        if conn: conn.close(); log_updates.append("DB connection closed.")

    return None, agent_log, None

def clarify_query(nl_query: str, schema_str: str, agent_log: list):

    nl_query_lower = nl_query.lower().strip()
    if nl_query_lower.startswith("get all ") or nl_query_lower.startswith("list all "):
        # Extract the table name
        table_name = nl_query_lower.split("all ", 1)[1].strip()
        # Remove any trailing words
        table_name = table_name.split(" ")[0]
        # Check if table_name exists in schema
        schema_tables = [line.split('"')[1].lower() for line in schema_str.splitlines() if line.startswith('CREATE TABLE')]
        if table_name in schema_tables:
            agent_log.append(f"Query '{nl_query}' is a clear 'get all' request for table '{table_name}'. Skipping clarification.")
            return None, agent_log, None

    """
    Analyzes the natural language query for ambiguity or underspecification using Gemini API.
    Returns clarifying questions if needed, or None if the query is clear.
    """
    try:
        clarification_prompt = """
        You are an expert in natural language query analysis for SQL generation.
        Your task is to analyze the user's natural language query against the provided database schema
        to identify ambiguity, underspecification, or missing details that could lead to incorrect SQL generation.
        If the query is clear and specific (e.g., 'get all circuits' or 'list drivers from 2018'), return an empty response.
        If the query is ambiguous or underspecified, return a JSON object with:
        - "needs_clarification": true
        - "clarifying_questions": a list of concise, user-friendly questions to resolve the ambiguity
        - "reason": a brief explanation of why clarification is needed

        ### Schema:
        {schema}

        ### Query:
        {query}

        ### Instructions:
        - Detect ambiguity (e.g., vague terms like 'recent', 'best', or missing table/column references).
        - Check for underspecification (e.g., missing conditions, time ranges, or specific entities).
        - Avoid flagging clear queries (e.g., 'get all circuits') for clarification unless truly ambiguous.
        - Return only a JSON object, nothing else.
        - Do not wrap the response in markdown (e.g., ```json). Return the raw JSON string.
        """
        prompt = clarification_prompt.format(schema=schema_str, query=nl_query)
        agent_log.append("Sending query to Gemini for clarification analysis.")

        response = gemini_model.generate_content(prompt)
        if not response or not response.candidates or not response.candidates[0].content.parts:
            agent_log.append("Error: No valid content in Gemini response.")
            return None, agent_log, "No valid content from Gemini clarification agent"

        clarification_text = response.candidates[0].content.parts[0].text.strip()
        agent_log.append(f"Raw Gemini response: {clarification_text}")

        # Clean the response by removing markdown formatting
        cleaned_text = clarification_text
        if cleaned_text.startswith("```json"):
            cleaned_text = cleaned_text.replace("```json", "", 1).strip()
        if cleaned_text.endswith("```"):
            cleaned_text = cleaned_text.rsplit("```", 1)[0].strip()

        agent_log.append(f"Cleaned Gemini response: {cleaned_text}")

        # Parse the JSON response
        try:
            result_json = json.loads(cleaned_text)
        except json.JSONDecodeError as e:
            agent_log.append(f"Error: Gemini response is not valid JSON: {cleaned_text}")
            # Fallback to assume query is clear if JSON parsing fails
            agent_log.append("Falling back to assume query is clear due to parsing failure.")
            return None, agent_log, None

        agent_log.append(f"Parsed Gemini clarification result: {result_json}")

        if result_json.get("needs_clarification", False):
            return result_json, agent_log, None
        else:
            return None, agent_log, None  # No clarification needed

    except Exception as e:
        error_msg = f"Error in clarification agent: {str(e)}"
        agent_log.append(error_msg)
        traceback.print_exc()
        return None, agent_log, error_msg



# --- Flask Routes ---

@app.route('/query', methods=['POST'])
def handle_query():
    agent_log = ["Processing /query request..."]
    MAX_OPTIMIZATION_ATTEMPTS = 2

    try:
        data = request.get_json()
        if not data or 'natural_language_query' not in data:
            return jsonify({"error": "Missing 'natural_language_query' in request body"}), 400

        nl_query = data['natural_language_query']
        agent_log.append(f"Received NL query: '{nl_query}'")

        # Get database schema
        schema_str, schema_log_updates = get_db_schema(DATABASE_FILE_PATH)
        agent_log.extend(schema_log_updates)
        if not schema_str:
            return jsonify({"error": "Failed to retrieve database schema.", "agent_log": agent_log}), 500

        # Step 1: Run clarification agent
        clarification_result, agent_log, clarification_error = clarify_query(nl_query, schema_str, agent_log)
        if clarification_error:
            return jsonify({"error": f"Clarification agent failed: {clarification_error}", "agent_log": agent_log}), 500

        if clarification_result and clarification_result.get("needs_clarification"):
            # Return clarifying questions to the user
            return jsonify({
                "message": "Query requires clarification",
                "natural_language_query": nl_query,
                "clarification": {
                    "questions": clarification_result.get("clarifying_questions", []),
                    "reason": clarification_result.get("reason", "")
                },
                "agent_log": agent_log
            }), 200

        # Step 2: Proceed with SQL generation if query is clear
        parsed_schema_dict = parse_schema_to_dict(schema_str, agent_log)

        raw_sql_from_hf, agent_log, hf_error = query_hf_endpoint(schema_str, nl_query, agent_log)
        if hf_error:
            return jsonify({"error": f"HF query failed: {hf_error}", "agent_log": agent_log,
                            "raw_response_from_hf": raw_sql_from_hf}), 500
        agent_log.append(f"Raw SQL from HF: '{raw_sql_from_hf if raw_sql_from_hf else 'None/Empty'}'")

        current_sql_for_optimizer = raw_sql_from_hf
        final_optimized_sql = None
        last_error_for_response = None
        validation_feedback_str = ""

        for attempt in range(MAX_OPTIMIZATION_ATTEMPTS):
            agent_log.append(
                f"Optimization attempt {attempt + 1}/{MAX_OPTIMIZATION_ATTEMPTS}. Feedback: '{validation_feedback_str if validation_feedback_str else 'None'}'")

            attempted_sql_opt, agent_log, opt_error = optimize_query_with_feedback(
                schema_str, nl_query, current_sql_for_optimizer, agent_log, validation_feedback_str
            )

            if opt_error or not attempted_sql_opt:
                agent_log.append(f"Optimizer failed/returned empty SQL (Attempt {attempt + 1}): {opt_error}")
                last_error_for_response = opt_error or "Optimizer returned no SQL"
                if attempt == MAX_OPTIMIZATION_ATTEMPTS - 1: break
                validation_feedback_str = "Previous optimization attempt failed to produce SQL. Please try again focusing on schema and question."
                current_sql_for_optimizer = ""  # Reset for next attempt
                continue

            agent_log.append(f"SQL from Optimizer (Attempt {attempt + 1}): {attempted_sql_opt}")
            final_optimized_sql = attempted_sql_opt

            if not parsed_schema_dict:
                agent_log.append("Skipping SQL validation: Parsed schema dictionary unavailable.")
                last_error_for_response = None  # Assume valid if cannot validate
                break

            validation_errors = validate_sql_against_schema(final_optimized_sql, parsed_schema_dict, agent_log)
            if not validation_errors:
                agent_log.append(f"SQL validation successful (Attempt {attempt + 1}).")
                last_error_for_response = None
                break
            else:
                agent_log.append(f"SQL validation errors (Attempt {attempt + 1}): {validation_errors}")
                last_error_for_response = f"SQL failed validation. Errors: {'; '.join(validation_errors)}"
                if attempt < MAX_OPTIMIZATION_ATTEMPTS - 1:
                    feedback_intro = "The previous SQL attempt had validation errors. Please fix them:"
                    validation_feedback_str = f"\n\n{feedback_intro}\n- " + "\n- ".join(validation_errors)
                    current_sql_for_optimizer = final_optimized_sql  # Try to fix this SQL

        if last_error_for_response:
            return jsonify({"error": f"Final SQL generation failed: {last_error_for_response}",
                            "agent_log": agent_log,
                            "final_attempted_sql": final_optimized_sql}), 500
        if not final_optimized_sql:
            return jsonify({"error": "Failed to produce any SQL query.", "agent_log": agent_log}), 500

        agent_log.append("Final SQL for execution determined.")

        query_results, column_names, db_error, exec_log_updates = execute_sql_query(DATABASE_FILE_PATH,
                                                                                    final_optimized_sql)
        agent_log.extend(exec_log_updates)
        if db_error:
            return jsonify({"error": f"DB execution error: {db_error}", "generated_sql": final_optimized_sql,
                            "agent_log": agent_log}), 500

        return jsonify({
            "message": "Query processed successfully",
            "natural_language_query": nl_query,
            "generated_sql": final_optimized_sql,
            "columns": column_names,
            "query_results": query_results,
            "agent_log": agent_log
        }), 200

    except Exception as e:
        error_msg = f"Unexpected error in /query: {str(e)}"
        agent_log.append(error_msg)
        agent_log.append(traceback.format_exc())
        print(error_msg)
        traceback.print_exc()
        return jsonify({"error": "Unexpected server error.", "agent_log": agent_log}), 500
@app.route('/')
def index():

    static_folder = os.path.join(os.path.dirname(__file__), 'static')
    if not os.path.exists(static_folder) or not os.path.exists(os.path.join(static_folder, 'index.html')):
        return "Error: Frontend not found.", 404
    return app.send_static_file('index.html')

if __name__ == "__main__":
    new_port = 5001
    print(f"INFO: Starting Flask app on port {new_port}")
    if not HF_API_URL or not HF_TOKEN:
        print("WARNING: Hugging Face config missing!")
    if not OPENAI_API_KEY or OPENAI_API_KEY == "your-openai-api-key":
        print("WARNING: OpenAI config missing or placeholder!")
    if not GEMINI_API_KEY or GEMINI_API_KEY == "your-gemini-api-key":
        print("WARNING: Gemini config missing or placeholder!")
    app.run(debug=True, port=new_port)