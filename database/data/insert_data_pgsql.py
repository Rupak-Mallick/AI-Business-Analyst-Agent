import psycopg2

# Database connection configuration
DB_CONFIG = {
    "host": "localhost",
    "port": 5433,
    "dbname": "ecommerce",
    "user": "postgres",
    "password": "password"
}

# Sample data to insert
products = [
    (1, "Laptop Pro", "High-performance laptop with 16GB RAM and 512GB SSD."),
    (2, "Smartphone X", "Latest smartphone with 5G support and 128GB storage."),
    (3, "Wireless Headphones", "Noise-cancelling headphones with 20-hour battery life."),
    (4, "Gaming Console", "Next-gen gaming console with 4K graphics."),
    (5, "Smart Watch", "Fitness tracker with heart rate monitor and GPS.")
]

sales = [
    (1, 1, 2, 999.99, "2025-01-15 10:00:00"),
    (2, 2, 5, 599.99, "2025-02-10 14:30:00"),
    (3, 3, 10, 149.99, "2025-03-05 09:15:00"),
    (4, 4, 3, 499.99, "2025-03-20 16:45:00"),
    (5, 5, 8, 199.99, "2025-04-10 11:00:00"),
    (6, 1, 1, 999.99, "2025-04-25 18:00:00"),
    (7, 3, 4, 149.99, "2025-05-01 08:30:00"),
    (8, 2, 2, 599.99, "2025-05-15 12:00:00"),
    (9, 5, 7, 199.99, "2025-05-22 15:30:00"),
    (10, 4, 1, 499.99, "2025-06-05 10:00:00")
]

def insert_data():
    """Inserts sample products and sales data into the database."""
    conn = None
    cur = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        print("✅ Connected to PostgreSQL. Creating tables and inserting data...")

        # Create products table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS products (
                product_id INT PRIMARY KEY,
                product_name TEXT NOT NULL,
                description TEXT
            );
        """)

        # Create sales table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS sales (
                id SERIAL PRIMARY KEY,
                order_id INT UNIQUE,
                product_id INT NOT NULL,
                quantity INT NOT NULL,
                unit_price DECIMAL(10, 2) NOT NULL,
                order_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (product_id) REFERENCES products (product_id) ON DELETE CASCADE
            );
        """)

        # Insert products
        cur.executemany("""
            INSERT INTO products (product_id, product_name, description)
            VALUES (%s, %s, %s)
            ON CONFLICT (product_id) DO NOTHING
        """, products)

        # Insert sales
        cur.executemany("""
            INSERT INTO sales (order_id, product_id, quantity, unit_price, order_date)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (order_id) DO NOTHING
        """, sales)

        conn.commit()
        print(f"✅ Inserted {len(products)} products and {len(sales)} sales successfully.")

    except Exception as e:
        print("❌ Error inserting data:", e)
        if conn:
            conn.rollback()

    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

if __name__ == "__main__":
    insert_data()
