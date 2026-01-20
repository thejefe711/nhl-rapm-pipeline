#!/usr/bin/env python3
"""
PostgreSQL Setup for Development

Help set up PostgreSQL for local development of SaaS hockey analytics.
"""

import subprocess
import sys
import os
from pathlib import Path

def check_postgres_installation():
    """Check if PostgreSQL is installed."""

    print("Checking PostgreSQL installation...")

    try:
        result = subprocess.run(['psql', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ PostgreSQL found: {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        pass

    print("❌ PostgreSQL not found")
    return False

def setup_postgres_service():
    """Help set up PostgreSQL service."""

    print("\nPostgreSQL Service Setup:")
    print("=" * 30)

    if sys.platform == "win32":
        print("Windows PostgreSQL Setup:")
        print("1. Download from: https://www.postgresql.org/download/windows/")
        print("2. Run installer as administrator")
        print("3. Set password for 'postgres' user")
        print("4. Add to PATH: C:\\Program Files\\PostgreSQL\\<version>\\bin")
        print("5. Start service: pg_ctl start -D \"C:\\Program Files\\PostgreSQL\\<version>\\data\"")

    elif sys.platform == "darwin":  # macOS
        print("macOS PostgreSQL Setup:")
        print("1. Install Homebrew: /bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"")
        print("2. Install PostgreSQL: brew install postgresql")
        print("3. Start service: brew services start postgresql")
        print("4. Create database: createdb hockey_analytics")

    else:  # Linux
        print("Linux PostgreSQL Setup:")
        print("Ubuntu/Debian:")
        print("  sudo apt update")
        print("  sudo apt install postgresql postgresql-contrib")
        print("  sudo systemctl start postgresql")
        print("  sudo -u postgres createuser --interactive --pwprompt")
        print("  sudo -u postgres createdb hockey_analytics")

def create_local_database():
    """Create local PostgreSQL database for development."""

    print("\nCreating Local Database:")
    print("=" * 25)

    commands = [
        "psql -U postgres -c \"CREATE DATABASE hockey_analytics;\"",
        "psql -U postgres -c \"CREATE USER hockey_user WITH PASSWORD 'hockey_pass';\"",
        "psql -U postgres -c \"GRANT ALL PRIVILEGES ON DATABASE hockey_analytics TO hockey_user;\""
    ]

    print("Run these commands:")
    for cmd in commands:
        print(f"  {cmd}")

    connection_string = "postgresql://hockey_user:hockey_pass@localhost:5432/hockey_analytics"
    print(f"\nSet DATABASE_URL environment variable:")
    print(f"  export DATABASE_URL=\"{connection_string}\"")

    # Create .env file
    env_file = Path(__file__).parent / ".env"
    with open(env_file, 'w') as f:
        f.write(f"DATABASE_URL={connection_string}\n")

    print(f"\nCreated .env file: {env_file}")

def cloud_database_options():
    """Show cloud database options."""

    print("\nCloud Database Options:")
    print("=" * 25)

    options = {
        "Supabase": {
            "url": "https://supabase.com",
            "pros": "Free tier, PostgreSQL, easy setup",
            "cons": "Limits on free tier",
            "cost": "$0-25/month"
        },
        "Neon": {
            "url": "https://neon.tech",
            "pros": "Serverless PostgreSQL, generous free tier",
            "cons": "Newer service",
            "cost": "$0-50/month"
        },
        "Railway": {
            "url": "https://railway.app",
            "pros": "Easy deployment, PostgreSQL included",
            "cons": "Limited free tier",
            "cost": "$0-10/month"
        },
        "AWS RDS": {
            "url": "https://aws.amazon.com/rds/",
            "pros": "Industry standard, scalable",
            "cons": "Complex setup, higher cost",
            "cost": "$20-100/month"
        }
    }

    for name, details in options.items():
        print(f"\n{name}:")
        print(f"  URL: {details['url']}")
        print(f"  Pros: {details['pros']}")
        print(f"  Cost: {details['cost']}")

def test_database_connection():
    """Test database connection."""

    print("\nTesting Database Connection:")
    print("=" * 30)

    test_script = '''
import os
import psycopg2

db_url = os.getenv('DATABASE_URL')
if not db_url:
    print("❌ DATABASE_URL not set")
    exit(1)

try:
    conn = psycopg2.connect(db_url)
    with conn.cursor() as cursor:
        cursor.execute("SELECT version();")
        version = cursor.fetchone()[0]
    conn.close()
    print(f"✅ Connected to PostgreSQL")
    print(f"Version: {version[:50]}...")
except Exception as e:
    print(f"❌ Connection failed: {e}")
'''

    print("Python test script:")
    print(test_script)

    # Save test script
    test_file = Path(__file__).parent / "test_db_connection.py"
    with open(test_file, 'w') as f:
        f.write(test_script)

    print(f"\nSaved test script: {test_file}")
    print("Run with: python test_db_connection.py")

def main():
    """Main setup function."""

    print("POSTGRESQL SETUP FOR HOCKEY ANALYTICS SAAS")
    print("=" * 50)

    if check_postgres_installation():
        print("✅ PostgreSQL is installed!")
    else:
        setup_postgres_service()
        return

    create_local_database()
    cloud_database_options()
    test_database_connection()

    print("\n" + "=" * 50)
    print("NEXT STEPS:")
    print("1. Set up PostgreSQL (local or cloud)")
    print("2. Set DATABASE_URL environment variable")
    print("3. Run: python database_migration.py")
    print("4. Test with: python test_db_connection.py")
    print("\nReady for SaaS database migration!")

if __name__ == "__main__":
    main()