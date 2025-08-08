"""
Comprehensive API Test Script for Medical RAG System
Tests all endpoints and provides detailed performance metrics
"""
import requests
import json
import time
from datetime import datetime
from typing import Dict, Any

# Configuration
BASE_URL = "http://localhost:8001"
TIMEOUT = 30

class APITester:
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.results = []
        
    def print_header(self, title: str):
        """Print a formatted header"""
        print("\n" + "="*80)
        print(f"🧪 {title}")
        print("="*80)
    
    def print_result(self, endpoint: str, status: str, response_time: float, details: str = ""):
        """Print test result"""
        status_emoji = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️"
        print(f"{status_emoji} {endpoint:<25} | {status:<6} | {response_time:>6.0f}ms | {details}")
    
    def test_endpoint(self, method: str, endpoint: str, data: Dict = None, expected_keys: list = None) -> Dict[str, Any]:
        """Test a single endpoint"""
        url = f"{self.base_url}{endpoint}"
        start_time = time.time()
        
        try:
            if method.upper() == "GET":
                response = requests.get(url, timeout=TIMEOUT)
            elif method.upper() == "POST":
                response = requests.post(url, json=data, timeout=TIMEOUT)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            response_time = (time.time() - start_time) * 1000
            
            # Check status code
            if response.status_code != 200:
                return {
                    "status": "FAIL",
                    "response_time": response_time,
                    "error": f"HTTP {response.status_code}",
                    "details": response.text[:100]
                }
            
            # Parse JSON response
            try:
                json_response = response.json()
            except json.JSONDecodeError:
                return {
                    "status": "FAIL",
                    "response_time": response_time,
                    "error": "Invalid JSON response",
                    "details": response.text[:100]
                }
            
            # Check expected keys
            if expected_keys:
                missing_keys = [key for key in expected_keys if key not in json_response]
                if missing_keys:
                    return {
                        "status": "PARTIAL",
                        "response_time": response_time,
                        "error": f"Missing keys: {missing_keys}",
                        "response": json_response
                    }
            
            return {
                "status": "PASS",
                "response_time": response_time,
                "response": json_response
            }
            
        except requests.exceptions.Timeout:
            return {
                "status": "FAIL",
                "response_time": TIMEOUT * 1000,
                "error": "Request timeout"
            }
        except requests.exceptions.ConnectionError:
            return {
                "status": "FAIL",
                "response_time": 0,
                "error": "Connection failed - Is server running?"
            }
        except Exception as e:
            return {
                "status": "FAIL",
                "response_time": (time.time() - start_time) * 1000,
                "error": str(e)
            }
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        self.print_header("HEALTH CHECK ENDPOINT")
        
        result = self.test_endpoint(
            "GET", 
            "/health",
            expected_keys=["status", "timestamp", "faiss_loaded", "total_documents"]
        )
        
        details = ""
        if result["status"] == "PASS":
            resp = result["response"]
            details = f"Status: {resp.get('status')}, FAISS: {resp.get('faiss_loaded')}, Docs: {resp.get('total_documents')}"
        else:
            details = result.get("error", "Unknown error")
        
        self.print_result("/health", result["status"], result["response_time"], details)
        self.results.append(("Health Check", result))
        return result
    
    def test_stats_endpoint(self):
        """Test system stats endpoint"""
        self.print_header("SYSTEM STATS ENDPOINT")
        
        result = self.test_endpoint(
            "GET",
            "/stats",
            expected_keys=["status", "faiss_loaded", "total_documents", "llm_model"]
        )
        
        details = ""
        if result["status"] == "PASS":
            resp = result["response"]
            details = f"Status: {resp.get('status')}, Model: {resp.get('llm_model', 'N/A')}"
        else:
            details = result.get("error", "Unknown error")
        
        self.print_result("/stats", result["status"], result["response_time"], details)
        self.results.append(("System Stats", result))
        return result
    
    def test_rag_endpoint(self):
        """Test main RAG endpoint"""
        self.print_header("RAG QUERY ENDPOINT")
        
        test_queries = [
            "What are the symptoms of epilepsy?",
            "How to treat seizures?",
            "What causes brain seizures?",
            "Hello, how are you?"  # Non-medical query
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n🔍 Test Query {i}: {query}")
            
            result = self.test_endpoint(
                "POST",
                "/rag",
                data={"query": query},
                expected_keys=["answer", "sources", "processing_time", "timestamp"]
            )
            
            details = ""
            if result["status"] == "PASS":
                resp = result["response"]
                answer_preview = resp.get("answer", "")[:50] + "..." if len(resp.get("answer", "")) > 50 else resp.get("answer", "")
                docs_retrieved = resp.get("documents_retrieved", "N/A")
                processing_time = resp.get("processing_time", "N/A")
                details = f"Docs: {docs_retrieved}, Process: {processing_time}s, Answer: {answer_preview}"
            else:
                details = result.get("error", "Unknown error")
            
            self.print_result(f"/rag (Query {i})", result["status"], result["response_time"], details)
            self.results.append((f"RAG Query {i}", result))
            
            # Small delay between requests
            time.sleep(0.5)
    
    def test_streaming_endpoint(self):
        """Test streaming RAG endpoint"""
        self.print_header("STREAMING RAG ENDPOINT")
        
        test_query = "What are seizure symptoms?"
        print(f"🔍 Streaming Test Query: {test_query}")
        
        url = f"{self.base_url}/rag/stream"
        start_time = time.time()
        
        try:
            response = requests.post(
                url,
                json={"query": test_query},
                stream=True,
                timeout=TIMEOUT
            )
            
            if response.status_code != 200:
                result = {
                    "status": "FAIL",
                    "response_time": (time.time() - start_time) * 1000,
                    "error": f"HTTP {response.status_code}"
                }
            else:
                # Process streaming response
                chunks_received = 0
                total_content = ""
                
                for line in response.iter_lines(decode_unicode=True):
                    if line.startswith("data: "):
                        chunks_received += 1
                        content = line[6:]  # Remove "data: " prefix
                        if content == "[DONE]":
                            break
                        try:
                            chunk_data = json.loads(content)
                            if "token" in chunk_data:
                                total_content += chunk_data["token"]
                        except json.JSONDecodeError:
                            pass
                
                response_time = (time.time() - start_time) * 1000
                
                if chunks_received > 0:
                    result = {
                        "status": "PASS",
                        "response_time": response_time,
                        "chunks": chunks_received,
                        "content_length": len(total_content)
                    }
                else:
                    result = {
                        "status": "FAIL",
                        "response_time": response_time,
                        "error": "No streaming chunks received"
                    }
            
        except Exception as e:
            result = {
                "status": "FAIL",
                "response_time": (time.time() - start_time) * 1000,
                "error": str(e)
            }
        
        details = ""
        if result["status"] == "PASS":
            details = f"Chunks: {result.get('chunks')}, Content: {result.get('content_length')} chars"
        else:
            details = result.get("error", "Unknown error")
        
        self.print_result("/rag/stream", result["status"], result["response_time"], details)
        self.results.append(("Streaming RAG", result))
        return result
    
    def test_docs_endpoint(self):
        """Test API documentation endpoint"""
        self.print_header("API DOCUMENTATION")
        
        # Test OpenAPI JSON
        result = self.test_endpoint("GET", "/openapi.json")
        
        details = ""
        if result["status"] == "PASS":
            resp = result["response"]
            paths_count = len(resp.get("paths", {}))
            details = f"OpenAPI spec with {paths_count} endpoints"
        else:
            details = result.get("error", "Unknown error")
        
        self.print_result("/openapi.json", result["status"], result["response_time"], details)
        self.results.append(("API Docs", result))
        return result
    
    def run_all_tests(self):
        """Run all endpoint tests"""
        print("🚀 Starting Comprehensive API Testing")
        print(f"📡 Target Server: {self.base_url}")
        print(f"⏰ Test Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Run all tests
        self.test_health_endpoint()
        self.test_stats_endpoint()
        self.test_rag_endpoint()
        self.test_streaming_endpoint()
        self.test_docs_endpoint()
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print test summary"""
        self.print_header("TEST SUMMARY")
        
        total_tests = len(self.results)
        passed_tests = sum(1 for _, result in self.results if result["status"] == "PASS")
        failed_tests = sum(1 for _, result in self.results if result["status"] == "FAIL")
        partial_tests = sum(1 for _, result in self.results if result["status"] == "PARTIAL")
        
        print(f"📊 Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"⚠️  Partial: {partial_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"📈 Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        # Calculate average response time for successful tests
        successful_times = [result["response_time"] for _, result in self.results if result["status"] == "PASS"]
        if successful_times:
            avg_response_time = sum(successful_times) / len(successful_times)
            print(f"⚡ Average Response Time: {avg_response_time:.0f}ms")
        
        # Print failed tests details
        if failed_tests > 0:
            print(f"\n❌ Failed Tests Details:")
            for test_name, result in self.results:
                if result["status"] == "FAIL":
                    print(f"   • {test_name}: {result.get('error', 'Unknown error')}")
        
        print(f"\n🎯 Overall Status: {'🎉 ALL SYSTEMS GO!' if failed_tests == 0 else '⚠️ NEEDS ATTENTION'}")
        print("="*80)

def main():
    """Main test function"""
    tester = APITester()
    tester.run_all_tests()

if __name__ == "__main__":
    main()
