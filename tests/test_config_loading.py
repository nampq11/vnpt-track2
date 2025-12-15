#!/usr/bin/env python3
"""Test script to verify per-model credentials loading"""

import os
import json
import tempfile
from pathlib import Path
from src.core.config import Config, VNPTConfig, VNPTModelCredentials


def test_json_config_loading():
    """Test loading credentials from JSON config file"""
    print("=" * 60)
    print("TEST 1: JSON Config File Loading")
    print("=" * 60)
    
    # Create temporary JSON config
    config_data = {
        "credentials": [
            {
                "authorization": "Bearer embedding_token_123",
                "tokenId": "embedding-token-id-uuid",
                "tokenKey": "embedding-token-key",
                "llmApiName": "LLM embeddings"
            },
            {
                "authorization": "Bearer small_token_456",
                "tokenId": "small-token-id-uuid",
                "tokenKey": "small-token-key",
                "llmApiName": "LLM small"
            },
            {
                "authorization": "Bearer large_token_789",
                "tokenId": "large-token-id-uuid",
                "tokenKey": "large-token-key",
                "llmApiName": "LLM large"
            }
        ]
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config_data, f)
        config_file = f.name
    
    try:
        # Set environment variable and create config
        os.environ['VNPT_CONFIG_FILE'] = config_file
        config = Config.from_env()
        
        # Verify credentials were loaded
        embedding_creds = config.vnpt.get_credentials("embedding")
        small_creds = config.vnpt.get_credentials("small")
        large_creds = config.vnpt.get_credentials("large")
        
        print(f"✓ Embedding credentials loaded: {bool(embedding_creds)}")
        print(f"  - API Key: {embedding_creds.api_key[:20]}...")
        print(f"  - Token ID: {embedding_creds.token_id}")
        
        print(f"✓ Small model credentials loaded: {bool(small_creds)}")
        print(f"  - API Key: {small_creds.api_key[:20]}...")
        print(f"  - Token ID: {small_creds.token_id}")
        
        print(f"✓ Large model credentials loaded: {bool(large_creds)}")
        print(f"  - API Key: {large_creds.api_key[:20]}...")
        print(f"  - Token ID: {large_creds.token_id}")
        
        print("\n✅ TEST 1 PASSED: JSON config file loading works\n")
        return True
    
    except Exception as e:
        print(f"\n❌ TEST 1 FAILED: {e}\n")
        return False
    
    finally:
        # Cleanup
        Path(config_file).unlink()
        if 'VNPT_CONFIG_FILE' in os.environ:
            del os.environ['VNPT_CONFIG_FILE']


def test_env_var_fallback():
    """Test fallback to environment variables"""
    print("=" * 60)
    print("TEST 2: Environment Variable Fallback")
    print("=" * 60)
    
    # Clear JSON config env var
    if 'VNPT_CONFIG_FILE' in os.environ:
        del os.environ['VNPT_CONFIG_FILE']
    
    # Set per-model environment variables
    os.environ['VNPT_API_KEY_EMBEDDING'] = 'Bearer embedding_env_token'
    os.environ['VNPT_TOKEN_ID_EMBEDDING'] = 'embedding-env-id'
    os.environ['VNPT_TOKEN_KEY_EMBEDDING'] = 'embedding-env-key'
    
    os.environ['VNPT_API_KEY_SMALL'] = 'Bearer small_env_token'
    os.environ['VNPT_TOKEN_ID_SMALL'] = 'small-env-id'
    os.environ['VNPT_TOKEN_KEY_SMALL'] = 'small-env-key'
    
    os.environ['VNPT_API_KEY_LARGE'] = 'Bearer large_env_token'
    os.environ['VNPT_TOKEN_ID_LARGE'] = 'large-env-id'
    os.environ['VNPT_TOKEN_KEY_LARGE'] = 'large-env-key'
    
    try:
        config = Config.from_env()
        
        # Verify credentials were loaded
        embedding_creds = config.vnpt.get_credentials("embedding")
        small_creds = config.vnpt.get_credentials("small")
        large_creds = config.vnpt.get_credentials("large")
        
        print(f"✓ Embedding credentials from env: {bool(embedding_creds)}")
        print(f"  - API Key: {embedding_creds.api_key}")
        print(f"  - Token ID: {embedding_creds.token_id}")
        
        print(f"✓ Small model credentials from env: {bool(small_creds)}")
        print(f"  - API Key: {small_creds.api_key}")
        print(f"  - Token ID: {small_creds.token_id}")
        
        print(f"✓ Large model credentials from env: {bool(large_creds)}")
        print(f"  - API Key: {large_creds.api_key}")
        print(f"  - Token ID: {large_creds.token_id}")
        
        print("\n✅ TEST 2 PASSED: Environment variable fallback works\n")
        return True
    
    except Exception as e:
        print(f"\n❌ TEST 2 FAILED: {e}\n")
        return False
    
    finally:
        # Cleanup
        for key in ['VNPT_API_KEY_EMBEDDING', 'VNPT_TOKEN_ID_EMBEDDING', 'VNPT_TOKEN_KEY_EMBEDDING',
                    'VNPT_API_KEY_SMALL', 'VNPT_TOKEN_ID_SMALL', 'VNPT_TOKEN_KEY_SMALL',
                    'VNPT_API_KEY_LARGE', 'VNPT_TOKEN_ID_LARGE', 'VNPT_TOKEN_KEY_LARGE']:
            if key in os.environ:
                del os.environ[key]


def test_legacy_credentials():
    """Test fallback to legacy single credential set"""
    print("=" * 60)
    print("TEST 3: Legacy Credentials Fallback")
    print("=" * 60)
    
    # Clear all per-model env vars
    for key in list(os.environ.keys()):
        if key.startswith('VNPT_') and 'EMBEDDING' in key or 'SMALL' in key or 'LARGE' in key:
            del os.environ[key]
    
    # Set legacy credentials
    os.environ['VNPT_API_KEY'] = 'Bearer legacy_token'
    os.environ['VNPT_TOKEN_ID'] = 'legacy-token-id'
    os.environ['VNPT_TOKEN_KEY'] = 'legacy-token-key'
    
    try:
        config = Config.from_env()
        
        # Verify credentials were loaded
        embedding_creds = config.vnpt.get_credentials("embedding")
        small_creds = config.vnpt.get_credentials("small")
        large_creds = config.vnpt.get_credentials("large")
        
        print(f"✓ Embedding credentials from legacy: {bool(embedding_creds)}")
        print(f"  - API Key: {embedding_creds.api_key}")
        
        print(f"✓ Small model credentials from legacy: {bool(small_creds)}")
        print(f"  - API Key: {small_creds.api_key}")
        
        print(f"✓ Large model credentials from legacy: {bool(large_creds)}")
        print(f"  - API Key: {large_creds.api_key}")
        
        # All should use the same legacy credentials
        assert embedding_creds.api_key == 'Bearer legacy_token'
        assert small_creds.api_key == 'Bearer legacy_token'
        assert large_creds.api_key == 'Bearer legacy_token'
        
        print("\n✅ TEST 3 PASSED: Legacy credentials fallback works\n")
        return True
    
    except Exception as e:
        print(f"\n❌ TEST 3 FAILED: {e}\n")
        return False
    
    finally:
        # Cleanup
        for key in ['VNPT_API_KEY', 'VNPT_TOKEN_ID', 'VNPT_TOKEN_KEY']:
            if key in os.environ:
                del os.environ[key]


def test_model_service_integration():
    """Test that services can be created with loaded credentials"""
    print("=" * 60)
    print("TEST 4: Service Integration")
    print("=" * 60)
    
    # Create temporary JSON config
    config_data = {
        "credentials": [
            {
                "authorization": "Bearer embedding_token_test",
                "tokenId": "embedding-id-test",
                "tokenKey": "embedding-key-test",
                "llmApiName": "LLM embeddings"
            },
            {
                "authorization": "Bearer small_token_test",
                "tokenId": "small-id-test",
                "tokenKey": "small-key-test",
                "llmApiName": "LLM small"
            },
            {
                "authorization": "Bearer large_token_test",
                "tokenId": "large-id-test",
                "tokenKey": "large-key-test",
                "llmApiName": "LLM large"
            }
        ]
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config_data, f)
        config_file = f.name
    
    try:
        os.environ['VNPT_CONFIG_FILE'] = config_file
        config = Config.from_env()
        
        # Try to create services (will validate credentials)
        from src.runtime.llm.vnpt_service import VNPTService
        from src.runtime.llm.embedding import VNPTEmbeddingService
        
        # Test VNPTService with small model
        try:
            service_small = VNPTService(config.vnpt, model_size="small")
            print(f"✓ VNPTService with small model created successfully")
        except ValueError as e:
            print(f"✗ Failed to create VNPTService with small model: {e}")
            return False
        
        # Test VNPTService with large model
        try:
            service_large = VNPTService(config.vnpt, model_size="large")
            print(f"✓ VNPTService with large model created successfully")
        except ValueError as e:
            print(f"✗ Failed to create VNPTService with large model: {e}")
            return False
        
        # Test VNPTEmbeddingService
        try:
            embedding_service = VNPTEmbeddingService(config.vnpt)
            print(f"✓ VNPTEmbeddingService created successfully")
        except ValueError as e:
            print(f"✗ Failed to create VNPTEmbeddingService: {e}")
            return False
        
        print("\n✅ TEST 4 PASSED: Services can be created with loaded credentials\n")
        return True
    
    except Exception as e:
        print(f"\n❌ TEST 4 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        Path(config_file).unlink()
        if 'VNPT_CONFIG_FILE' in os.environ:
            del os.environ['VNPT_CONFIG_FILE']


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("VNPT Per-Model Credentials Configuration Tests")
    print("=" * 60 + "\n")
    
    results = []
    
    results.append(("JSON Config Loading", test_json_config_loading()))
    results.append(("Environment Variable Fallback", test_env_var_fallback()))
    results.append(("Legacy Credentials", test_legacy_credentials()))
    results.append(("Service Integration", test_model_service_integration()))
    
    # Summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    total_passed = sum(1 for _, passed in results if passed)
    total_tests = len(results)
    
    print(f"\nTotal: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total_tests - total_passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit(main())

