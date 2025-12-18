import psycopg2

def get_connection():
    return psycopg2.connect(
        host="avo-adb-001.postgres.database.azure.com",
        port=5432,
        database="ChatGPT_DB",   
        user="adminavo",     
        password="$#fKcdXPg4@ue8AW",
        sslmode="require"
    )
def get_connection_sales():
    return psycopg2.connect(
        host="avo-adb-001.postgres.database.azure.com",
        port=5432,
        database="Sales_DB",   
        user="adminavo",     
        password="$#fKcdXPg4@ue8AW",
        sslmode="require"
    )

