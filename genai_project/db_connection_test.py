import sqlite3 # Python's built-in library for SQLite

def query_formula_1_database(db_path, sql_query):
    """
    Connects to the SQLite database, executes a query, and prints the results.
    """
    try:
        # Establish a connection to the SQLite database
        conn = sqlite3.connect(db_path)
        print(f"Successfully connected to database: {db_path}")

        # Create a cursor object to execute SQL queries
        cursor = conn.cursor()

        # Execute the SQL query
        print(f"\nExecuting query: {sql_query}")
        cursor.execute(sql_query)

        # Fetch all the results
        results = cursor.fetchall()

        if results:
            print("\nQuery Results:")
            # Print column names (optional, but helpful)
            col_names = [description[0] for description in cursor.description]
            print(col_names)
            # Print each row
            for row in results:
                print(row)
        else:
            print("No results returned for this query.")

    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Close the connection if it was established
        if 'conn' in locals() and conn:
            conn.close()
            print("\nDatabase connection closed.")

# --- Main part of the script ---
if __name__ == "__main__":

    database_file_path = "./formula_1/formula_1.sqlite"

    example_query = "SELECT circuitId, name, location, country FROM circuits LIMIT 5;"
    # Another example: "SELECT driverRef, forename, surname, nationality FROM drivers LIMIT 5;"

    query_formula_1_database(database_file_path, example_query)