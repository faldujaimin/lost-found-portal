import sqlite3
import os
import json

db_path = os.path.join('instance', 'site.db')
conn = sqlite3.connect(db_path)
cursor = conn.cursor()
cursor.execute("SELECT id, item_name, image_filename, verification_score, verification_details FROM item")
rows = cursor.fetchall()
for row in rows:
    if row[1] == 'bag' or row[0] == 140 or 'bag' in row[1].lower():
        print(f"ID: {row[0]}, Name: {row[1]}, Image: {row[2]}, Score: {row[3]}")
        print(f"Details: {row[4]}")
conn.close()
