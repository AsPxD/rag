"""
Migration script to move from old app.py structure to optimized API structure
This script will help preserve your existing FAISS index and documents
"""
import os
import shutil
import pickle
from pathlib import Path

def migrate_faiss_data():
    """Migrate existing FAISS data to new structure"""
    print("🔄 Migrating FAISS data...")
    
    old_faiss_path = "faiss_index"
    
    if not os.path.exists(old_faiss_path):
        print("❌ No existing FAISS index found. Please ensure your FAISS index is built first.")
        return False
    
    # Check for required files
    index_file = os.path.join(old_faiss_path, "index.faiss")
    docs_file = os.path.join(old_faiss_path, "index.pkl")
    
    if not os.path.exists(index_file):
        print(f"❌ FAISS index file not found at {index_file}")
        return False
    
    if os.path.exists(docs_file):
        # Rename index.pkl to docs.pkl for consistency
        new_docs_file = os.path.join(old_faiss_path, "docs.pkl")
        if not os.path.exists(new_docs_file):
            shutil.copy2(docs_file, new_docs_file)
            print(f"✅ Copied {docs_file} to {new_docs_file}")
    
    print("✅ FAISS data migration completed!")
    return True

def create_env_template():
    """Create .env template if it doesn't exist"""
    env_file = ".env"
    
    if not os.path.exists(env_file):
        with open(env_file, "w") as f:
            f.write("# Google API Key for Gemini 2.5 Pro\n")
            f.write("GOOGLE_API_KEY=your_google_api_key_here\n")
            f.write("\n# Optional: Custom port (default: 8000)\n")
            f.write("PORT=8000\n")
        print("✅ Created .env template")
    else:
        print("✅ .env file already exists")

def backup_old_files():
    """Backup old files"""
    backup_dir = "backup_old"
    os.makedirs(backup_dir, exist_ok=True)
    
    files_to_backup = ["app.py", "frontend.py"]
    
    for file in files_to_backup:
        if os.path.exists(file):
            shutil.copy2(file, os.path.join(backup_dir, file))
            print(f"✅ Backed up {file}")

def main():
    print("🚀 Starting migration to optimized RAG structure...")
    print("=" * 60)
    
    # Step 1: Backup old files
    print("\n📦 Step 1: Backing up old files...")
    backup_old_files()
    
    # Step 2: Migrate FAISS data
    print("\n🔄 Step 2: Migrating FAISS data...")
    if not migrate_faiss_data():
        print("❌ Migration failed. Please check your FAISS index.")
        return
    
    # Step 3: Create environment template
    print("\n⚙️  Step 3: Setting up environment...")
    create_env_template()
    
    print("\n" + "=" * 60)
    print("✅ Migration completed successfully!")
    print("\n📋 Next steps:")
    print("1. Update your .env file with your actual GOOGLE_API_KEY")
    print("2. Test the new API: cd api && python main.py")
    print("3. Deploy to Render using Dockerfile or Vercel using vercel.json")
    print("\n🔗 New API endpoints:")
    print("  • Health: GET /health")
    print("  • RAG: POST /rag")
    print("  • Streaming: POST /rag/stream")
    print("  • Stats: GET /stats")

if __name__ == "__main__":
    main()
