import altibase_driver

# Same connection string format as Go driver
conn_str = "Server=127.0.0.1;User=SYS;Password=MANAGER;PORT=20300"

# Connect and execute query
conn = altibase_driver.connect(conn_str)
cursor = conn.cursor()

# Execute the same query as Go example
cursor.execute("SELECT SYSDATE FROM DUAL")
results = cursor.fetchall()

for row in results:
    print(f"Time: {row[0]}")

conn.close()