import mysql.connector
from mysql.connector import Error
from char_extraction import main
from arduino_signal import send_open_signal

def check_plate_in_database(plate_number):

    db_config = {
        "database": "allowed_db",
        "user": "root",
        "password": "shreyank",
        "host": "localhost",
        "port": "3306"
    }

    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        query = "SELECT EXISTS(SELECT 1 FROM allowed_vehicles WHERE allowed_number = %s)"
        cursor.execute(query, (plate_number,))
        exists = cursor.fetchone()[0]

        cursor.close()
        conn.close()

        return exists

    except Error as e:
        print("Database error:", e)
        return False

if __name__ == "__main__":

    plate_number = main("", "")

    if plate_number and "No" not in plate_number:
        print(f"Recognized License Plate: {plate_number}")
        if check_plate_in_database(plate_number):
            print("Plate found in database. Sending signal to Arduino...")
            send_open_signal()
        else:
            print("Plate not found in database. Access denied.")
    else:
        print("License plate could not be recognized.")
