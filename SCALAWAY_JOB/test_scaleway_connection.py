#!/usr/bin/env python3
"""
Test script to verify Scalaway S3 connection and list objects
"""

import os
from dotenv import load_dotenv
from pipeline.storage import storage_from_env

# Load environment variables
load_dotenv('vm.env')

def test_scaleway_connection():
    """Test connection to Scalaway S3"""
    print("🔌 Testing Scalaway S3 connection...")

    try:
        # Create storage client
        storage_client = storage_from_env()
        print(f"✅ Storage client created for bucket: {storage_client.bucket}")

        # Test listing objects at root level
        print("📋 Testing list_objects at root level...")
        try:
            objects = storage_client.list_objects("")
            print(f"✅ Found {len(objects)} objects at root level")
            if objects:
                print("   Sample objects:")
                for obj in objects[:5]:  # Show first 5
                    print(f"   - {obj}")
        except Exception as e:
            print(f"⚠️  Could not list root objects: {e}")

        # Test listing objects in the target directory
        target_prefix = "soil-sentinel_batch_results_2015_2018/aggregated/"
        print(f"📋 Testing list_objects for: {target_prefix}")
        try:
            objects = storage_client.list_objects(target_prefix)
            print(f"✅ Found {len(objects)} objects in {target_prefix}")
            if objects:
                print("   Sample objects:")
                for obj in objects[:5]:  # Show first 5
                    print(f"   - {obj}")
            else:
                print(f"   📂 Directory {target_prefix} exists but is empty")
        except Exception as e:
            print(f"⚠️  Could not list objects in {target_prefix}: {e}")
            print(f"   This likely means the directory doesn't exist yet")

        # Test getting a sample object (if any exist)
        if objects:
            sample_obj = objects[0]
            print(f"📥 Testing get_text for: {sample_obj}")
            try:
                content = storage_client.get_text(sample_obj)
                print(f"✅ Successfully retrieved {len(content)} characters")
                print("   Sample content:")
                print("   " + "\n   ".join(content.splitlines()[:3]) + "...")
            except Exception as e:
                print(f"⚠️  Could not get object content: {e}")

        print("\n✅ Scalaway S3 connection test completed!")

    except Exception as e:
        print(f"❌ Failed to create storage client: {e}")
        return False

    return True

if __name__ == "__main__":
    test_scaleway_connection()