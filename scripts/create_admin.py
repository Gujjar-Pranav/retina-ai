from pathlib import Path
import json
import hashlib

ROOT = Path(__file__).resolve().parents[1]
USERS_FILE = ROOT / "data" / "users.json"

USERS_FILE.parent.mkdir(parents=True, exist_ok=True)

def hash_pw(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

admin_user = {
    "user_id": "admin",
    "password_hash": hash_pw("admin123"),  # 👈 CHANGE AFTER LOGIN
    "role": "admin",
    "active": True,
}

if USERS_FILE.exists():
    users = json.loads(USERS_FILE.read_text())
else:
    users = {}

users["admin"] = admin_user
USERS_FILE.write_text(json.dumps(users, indent=2))

print("✅ Admin user created")
print("User ID: admin")
print("Password: admin123")
