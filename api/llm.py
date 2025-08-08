"""
LLM abstraction layer for easy model switching
Currently using Gemini 2.5 Pro with streaming support
"""
import os
import asyncio
from typing import AsyncIterator
import google.generativeai as genai
from abc import ABC, abstractmethod

class BaseLLM(ABC):
    """Abstract base class for LLM implementations"""
    
    @abstractmethod
    async def stream_chat(self, system: str, user: str) -> AsyncIterator[str]:
        """Stream chat completion tokens"""
        pass

class GeminiProLLM(BaseLLM):
    """Gemini 2.5 Pro implementation with streaming"""
    
    def __init__(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set in environment variables")
        
        genai.configure(api_key=api_key)
        # Use the correct stable model name
        self.model = genai.GenerativeModel("gemini-2.5-pro")
        print("✅ Gemini 2.5 Pro model initialized successfully")
        
    async def stream_chat(self, system: str, user: str) -> AsyncIterator[str]:
        """Stream chat completion with system instruction"""
        try:
            # Create a focused medical prompt
            full_prompt = f"""{system}

Context: Use the provided medical information to answer the question accurately and concisely.

Question: {user}

Answer:"""
            
            print(f"🤖 Generating response for: {user[:50]}...")
            
            # Generate streaming response
            def _generate_content():
                return self.model.generate_content(
                    full_prompt,
                    stream=True,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.3,  # Lower temperature for more focused responses
                        max_output_tokens=800,
                        top_p=0.8,
                        top_k=40
                    )
                )
            
            response = await asyncio.to_thread(_generate_content)
            
            # Process streaming chunks
            token_count = 0
            for chunk in response:
                try:
                    if hasattr(chunk, 'text') and chunk.text:
                        token_count += 1
                        yield chunk.text
                    elif hasattr(chunk, 'parts') and chunk.parts:
                        for part in chunk.parts:
                            if hasattr(part, 'text') and part.text:
                                token_count += 1
                                yield part.text
                except Exception as chunk_error:
                    print(f"Chunk processing error: {chunk_error}")
                    continue
            
            print(f"✅ Generated {token_count} tokens successfully")
                    
        except Exception as e:
            error_msg = str(e)
            print(f"❌ LLM Error: {error_msg}")
            
            # Provide more specific error messages
            if "API_KEY" in error_msg.upper():
                yield "Error: Invalid or missing Google API key. Please check your configuration."
            elif "QUOTA" in error_msg.upper() or "LIMIT" in error_msg.upper():
                yield "Error: API quota exceeded. Please try again later."
            elif "PERMISSION" in error_msg.upper():
                yield "Error: API access denied. Please check your API key permissions."
            else:
                yield f"I apologize, but I encountered an error while generating the response. Please try again. (Error: {error_msg[:100]})"
            
            # Also yield a helpful message
            yield "\n\nIf this error persists, please verify your Google API key is correctly set in the .env file."

# Factory function for easy model switching
def get_llm() -> BaseLLM:
    """Get the configured LLM instance"""
    return GeminiProLLM()
