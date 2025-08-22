

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai
import google.generativeai as genai
import os
from typing import Optional, List
import asyncio
import uvicorn
from datetime import datetime
import logging
from dotenv import load_dotenv
import requests
import re
import razorpay
import hmac
import hashlib
import json

# Configure logging
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_environment():
    """Validate required environment variables"""
    required_vars = [
        "RAZORPAY_KEY_ID",
        "RAZORPAY_KEY_SECRET"
    ]
    
    recommended_vars = [
        "OPENAI_API_KEY",
        "GEMINI_API_KEY"
    ]
    
    missing_required = [var for var in required_vars if not os.getenv(var)]
    missing_recommended = [var for var in recommended_vars if not os.getenv(var)]
    
    if missing_required:
        logger.error(f"Missing required environment variables: {missing_required}")
        raise ValueError(f"Required environment variables not set: {missing_required}")
    
    if missing_recommended:
        logger.warning(f"Missing recommended environment variables: {missing_recommended}")
        logger.warning("Some AI features may not work properly")


app = FastAPI(title="AI Book Generator API", version="2.2.0")

validate_environment()  # Uncomment this line if you want strict validation


# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8080",
        "https://book.verstiileai.com",  # Add your actual frontend URL
        # Add more origins as needed
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Initialize AI clients
try:
    openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client: {e}")
    openai_client = None

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize Razorpay client
try:
    razorpay_client = razorpay.Client(
        auth=(os.getenv("RAZORPAY_KEY_ID"), os.getenv("RAZORPAY_KEY_SECRET"))
    )
except Exception as e:
    logger.error(f"Failed to initialize Razorpay client: {e}")
    razorpay_client = None

# Pydantic models
class ImageConfig(BaseModel):
    include_images: bool = False
    image_count: int = 3
    image_style: str = "professional"
    image_source: str = "generate"

class GenerateRequest(BaseModel):
    prompt: str
    provider: str = "openai"
    max_tokens: int = 4000
    temperature: float = 0.7
    generate_cover: bool = True
    cover_style: str = "professional"
    image_config: Optional[ImageConfig] = None

class WalletRechargeRequest(BaseModel):
    amount: int
    user_id: str
    user_profile: dict

class PaymentVerificationRequest(BaseModel):
    razorpay_order_id: str
    razorpay_payment_id: str
    razorpay_signature: str
    user_id: str
    amount: int

class BookGenerationRequest(BaseModel):
    user_id: str
    book_params: dict
    user_profile: dict

class ImageData(BaseModel):
    url: str
    caption: str
    source: str
    alt_text: str

class CoverImage(BaseModel):
    url: str
    prompt_used: str
    style: str

class HealthResponse(BaseModel):
    message: str
    timestamp: str
    status: str

class GenerateResponse(BaseModel):
    content: str
    provider: str
    timestamp: str
    word_count: int
    cover_image: Optional[CoverImage] = None
    images: List[ImageData] = []

class PaymentOrderResponse(BaseModel):
    order_id: str
    amount: int
    currency: str
    key_id: str
    success: bool

class PaymentVerificationResponse(BaseModel):
    success: bool
    message: str
    transaction_id: Optional[str] = None

# In-memory storage for transactions (replace with your database)
transactions_db = {}

def clean_content(content: str) -> str:
    """Clean and filter content to keep only book-relevant text"""
    
    # Remove common non-book content patterns
    patterns_to_remove = [
        r'I hope this helps.*?(?=\n|\.|$)',
        r'If you need.*?(?=\n|\.|$)',
        r'Please let me know.*?(?=\n|\.|$)',
        r'Feel free to.*?(?=\n|\.|$)',
        r'I\'d be happy to.*?(?=\n|\.|$)',
        r'Would you like me to.*?(?=\n|\.|$)',
        r'Let me know if you.*?(?=\n|\.|$)',
        r'Is there anything else.*?(?=\n|\.|$)',
        r'Here\'s.*?chapter.*?:?\s*',
        r'This chapter.*?(?=\n)',
        r'As requested.*?(?=\n)',
        r'Based on your.*?(?=\n)',
        r'I\'ve created.*?(?=\n)',
        r'I\'ll help.*?(?=\n)',
        r'Here\'s what I.*?(?=\n)',
        r'^(Sure|Certainly|Of course|Absolutely)[\s\.,!]+',
        r'Hope this.*?(?=\n|\.|$)',
        r'This should.*?(?=\n|\.|$)',
        r'I think.*?(?=\n|\.|$)',
        r'Note that.*?(?=\n|\.|$)',
        r'Please note.*?(?=\n|\.|$)',
        r'Important note.*?(?=\n|\.|$)',
        r'\*\*Note:.*?\*\*',
        r'\*Note:.*?\*',
        r'Disclaimer:.*?(?=\n)',
        r'\(Note:.*?\)',
        r'AI Assistant:.*?(?=\n)',
        r'As an AI.*?(?=\n)',
        r'I should mention.*?(?=\n)',
        r'It\'s worth noting.*?(?=\n)',
    ]
    
    # Apply cleaning patterns
    cleaned_content = content
    for pattern in patterns_to_remove:
        cleaned_content = re.sub(pattern, '', cleaned_content, flags=re.IGNORECASE | re.MULTILINE)
    
    # Remove excessive whitespace
    cleaned_content = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned_content)
    cleaned_content = re.sub(r'^\s+', '', cleaned_content, flags=re.MULTILINE)
    
    # Remove lines that are clearly assistant responses
    lines = cleaned_content.split('\n')
    filtered_lines = []
    
    skip_patterns = [
        r'^(Here\'s|Here is)',
        r'^I\'ve (created|written|prepared)',
        r'^This (chapter|section|content)',
        r'^Let me (know|help)',
        r'^Would you like',
        r'^If you (need|want|would like)',
        r'^Feel free to',
        r'^Please (let me know|feel free)',
        r'^I hope this',
        r'^Certainly',
        r'^Of course',
        r'^Sure',
        r'^Absolutely',
    ]
    
    for line in lines:
        line = line.strip()
        if not line:
            filtered_lines.append('')
            continue
            
        should_skip = False
        for pattern in skip_patterns:
            if re.match(pattern, line, re.IGNORECASE):
                should_skip = True
                break
        
        if not should_skip:
            filtered_lines.append(line)
    
    # Join lines and clean up again
    cleaned_content = '\n'.join(filtered_lines)
    cleaned_content = re.sub(r'\n{3,}', '\n\n', cleaned_content)
    
    return cleaned_content.strip()

def extract_pure_book_content(content: str) -> str:
    """Extract only the pure book content, removing all assistant commentary"""
    
    # First apply general cleaning
    cleaned = clean_content(content)
    
    # Look for chapter markers and content structure
    lines = cleaned.split('\n')
    book_lines = []
    
    # Track if we're in actual book content
    in_book_content = False
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines but preserve them in book content
        if not line:
            if in_book_content:
                book_lines.append('')
            continue
        
        # Check if this looks like a chapter heading
        if re.match(r'^(Chapter \d+|#|##)', line, re.IGNORECASE):
            in_book_content = True
            book_lines.append(line)
            continue
        
        # Check if this looks like a section heading
        if re.match(r'^#{1,6}\s+', line) or re.match(r'^[A-Z][A-Za-z\s:]{10,60}$', line):
            in_book_content = True
            book_lines.append(line)
            continue
        
        # If we're in book content, include the line unless it's clearly assistant text
        if in_book_content:
            # Skip lines that are clearly not book content
            if not re.match(r'^(Here|I\'ve|This|Let me|Would you|If you|Feel free|Please|I hope|Certainly|Of course|Sure|Absolutely)', line, re.IGNORECASE):
                book_lines.append(line)
        else:
            # Check if this line starts actual book content (not assistant text)
            if (len(line) > 20 and 
                not re.match(r'^(Here|I\'ve|This|Let me|Would you|If you|Feel free|Please|I hope|Certainly|Of course|Sure|Absolutely)', line, re.IGNORECASE) and
                not line.endswith('?') and
                '.' in line):
                in_book_content = True
                book_lines.append(line)
    
    result = '\n'.join(book_lines)
    
    # Final cleanup
    result = re.sub(r'\n{3,}', '\n\n', result)
    
    return result.strip()

def verify_payment_signature(order_id: str, payment_id: str, signature: str) -> bool:
    """Verify Razorpay payment signature"""
    try:
        # Create the message to be verified
        message = f"{order_id}|{payment_id}"
        
        # Generate expected signature
        expected_signature = hmac.new(
            os.getenv("RAZORPAY_KEY_SECRET", "").encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return hmac.compare_digest(signature, expected_signature)
    except Exception as e:
        logger.error(f"Signature verification failed: {str(e)}")
        return False

# @app.get("/api/health", response_model=HealthResponse)
# async def health_check():
#     """Health check endpoint"""
#     return HealthResponse(
#         message="AI Book Generator Backend is running!",
#         timestamp=datetime.now().isoformat(),
#         status="healthy"
#     )
@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Enhanced health check endpoint"""
    
    # Check AI providers availability
    providers_status = {
        "openai": bool(openai_client and os.getenv("OPENAI_API_KEY")),
        "gemini": bool(os.getenv("GEMINI_API_KEY")),
        "razorpay": bool(razorpay_client and os.getenv("RAZORPAY_KEY_ID") and os.getenv("RAZORPAY_KEY_SECRET"))
    }
    
    # Overall status
    critical_services = ["razorpay"]  # Services that must be available
    ai_services = ["openai", "gemini"]  # At least one must be available
    
    critical_ok = all(providers_status[service] for service in critical_services)
    ai_ok = any(providers_status[service] for service in ai_services)
    
    status = "healthy" if (critical_ok and ai_ok) else "degraded"
    
    return HealthResponse(
        message="AI Book Generator Backend is running!",
        timestamp=datetime.now().isoformat(),
        status=status,
        # Note: HealthResponse model doesn't include providers_status
        # You can add it to the model if you want this info
    )
    
    
@app.post("/api/create-order", response_model=PaymentOrderResponse)
async def create_payment_order(request: WalletRechargeRequest):
    """Create a Razorpay order for wallet recharge"""
    try:
        if not razorpay_client:
            raise HTTPException(status_code=500, detail="Payment gateway not configured")
        
        # Validate amount (minimum ₹100)
        if request.amount < 1:
            raise HTTPException(status_code=400, detail="Minimum recharge amount is ₹100")
        
        # Create order
        order_data = {
            "amount": request.amount * 100,  # Convert to paise
            "currency": "INR",
            "receipt": f"wallet_recharge",
            "notes": {
                "user_id": request.user_id,
                "purpose": "wallet_recharge"
            }
        }
        
        order = razorpay_client.order.create(data=order_data)
        
        # Store transaction details temporarily
        transaction_id = f"txn_{request.user_id}_{int(datetime.now().timestamp())}"
        transactions_db[order['id']] = {
            "transaction_id": transaction_id,
            "user_id": request.user_id,
            "amount": request.amount,
            "type": "credit",
            "status": "pending",
            "created_at": datetime.now().isoformat()
        }
        
        return PaymentOrderResponse(
            order_id=order['id'],
            amount=order['amount'],
            currency=order['currency'],
            key_id=os.getenv("RAZORPAY_KEY_ID", ""),
            success=True
        )
        
    except Exception as e:
        logger.error(f"Order creation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create order: {str(e)}")

@app.post("/api/payment/verify", response_model=PaymentVerificationResponse)
async def verify_payment(request: PaymentVerificationRequest):
    """Verify payment and update wallet balance"""
    try:
        # Verify payment signature
        if not verify_payment_signature(
            request.razorpay_order_id,
            request.razorpay_payment_id,
            request.razorpay_signature
        ):
            raise HTTPException(status_code=400, detail="Invalid payment signature")
        
        # Get transaction details
        transaction = transactions_db.get(request.razorpay_order_id)
        if not transaction:
            raise HTTPException(status_code=404, detail="Transaction not found")
        
        # Update transaction status
        transaction["status"] = "completed"
        transaction["payment_id"] = request.razorpay_payment_id
        transaction["completed_at"] = datetime.now().isoformat()
        
        # Here you would update your database with the new wallet balance
        # For now, we'll just return success
        # In a real implementation, you'd:
        # 1. Update user's wallet balance in your database
        # 2. Save the transaction record
        # 3. Send confirmation email/SMS
        
        logger.info(f"Payment verified successfully for user {request.user_id}, amount: ₹{request.amount}")
        
        return PaymentVerificationResponse(
            success=True,
            message=f"₹{request.amount} has been added to your wallet successfully!",
            transaction_id=transaction["transaction_id"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Payment verification failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Payment verification failed: {str(e)}")
    
@app.on_event("startup")
async def startup_event():
    """Run startup tasks"""
    logger.info("🚀 AI Book Generator Backend starting up...")
    
    # Validate environment variables
    try:
        validate_environment()
        logger.info("✅ Environment validation passed")
    except Exception as e:
        logger.error(f"❌ Environment validation failed: {e}")
    
    # Test AI providers
    if openai_client:
        logger.info("✅ OpenAI client initialized")
    else:
        logger.warning("⚠️ OpenAI client not available")
    
    if os.getenv("GEMINI_API_KEY"):
        logger.info("✅ Gemini API key available")
    else:
        logger.warning("⚠️ Gemini API key not available")
    
    if razorpay_client:
        logger.info("✅ Razorpay client initialized")
    else:
        logger.error("❌ Razorpay client not available")
    
    logger.info("🎉 Startup completed!")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean shutdown"""
    logger.info("👋 AI Book Generator Backend shutting down...")

@app.post("/api/payment/book-generation")
async def process_book_generation_payment(request: BookGenerationRequest):
    """Process payment for book generation using wallet balance"""
    try:
        # Calculate costs
        length_costs = {
            'xxs': 99,
          'xs': 129,
          's': 149,
          'm': 169,
          'l': 199,
          'xl': 229,
          'xxl': 249
        }
        
        cover_cost = 79 if request.book_params.get('generate_cover', False) else 0
        total_cost = length_costs.get(request.book_params.get('length', 'short'), 149) + cover_cost
        
        # Check wallet balance
        current_balance = request.user_profile.get('wallet_balance', 0)
        if current_balance < total_cost:
            raise HTTPException(
                status_code=400, 
                detail=f"Insufficient wallet balance. Required: ₹{total_cost}, Available: ₹{current_balance}"
            )
        
        # Here you would:
        # 1. Deduct amount from user's wallet in your database
        # 2. Create a debit transaction record
        # 3. Call the book generation function
        # 4. Return the generated book details
        
        # For now, creating a mock response
        transaction_id = f"book_txn_{request.user_id}_{int(datetime.now().timestamp())}"
        
        # Store transaction details
        transactions_db[transaction_id] = {
            "transaction_id": transaction_id,
            "user_id": request.user_id,
            "amount": total_cost,
            "type": "debit",
            "description": f"Book generation - {request.book_params.get('length', 'short')} book",
            "status": "completed",
            "created_at": datetime.now().isoformat()
        }
        
        # Generate book (you would call your existing generate_content function here)
        book_content = await generate_book_content(request.book_params)
        
        return {
            "success": True,
            "message": "Book generated successfully!",
            "transaction_id": transaction_id,
            "book_content": book_content,
            "cost": total_cost,
            "new_balance": current_balance - total_cost
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Book generation payment failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Book generation failed: {str(e)}")

async def generate_book_content(book_params: dict) -> dict:
    """Generate book content based on parameters"""
    try:
        # Create a GenerateRequest from book_params
        generate_request = GenerateRequest(
            prompt=book_params.get('prompt', ''),
            provider=book_params.get('provider', 'openai'),
            generate_cover=book_params.get('generate_cover', True),
            cover_style=book_params.get('cover_style', 'professional')
        )
        
        # Generate content using existing function
        response = await generate_content(generate_request)
        
        return {
            "content": response.content,
            "word_count": response.word_count,
            "cover_image": response.cover_image,
            "provider": response.provider
        }
        
    except Exception as e:
        logger.error(f"Book content generation failed: {str(e)}")
        raise e

@app.get("/api/payment/config")
async def get_payment_config():
    """Get payment configuration"""
    return {
        "razorpay_key_id": os.getenv("RAZORPAY_KEY_ID", ""),
        "currency": "INR",
        "company_name": "BookForge AI",
        "min_recharge": 1,
        "max_recharge": 50000,
        "book_prices": {
             'xxs': 99,
          'xs': 129,
          's': 149,
          'm': 169,
          'l': 199,
          'xl': 229,
          'xxl': 249,
            "cover": 79
        }
    }

@app.get("/api/transactions/{user_id}")
async def get_user_transactions(user_id: str):
    """Get user transaction history"""
    try:
        # Filter transactions for the user
        user_transactions = []
        for order_id, transaction in transactions_db.items():
            if transaction.get("user_id") == user_id:
                user_transactions.append({
                    "id": transaction.get("transaction_id", order_id),
                    **transaction
                })
        
        # Sort by created_at descending
        user_transactions.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        
        return {
            "transactions": user_transactions[:10],  # Return last 10 transactions
            "total_count": len(user_transactions)
        }
        
    except Exception as e:
        logger.error(f"Failed to fetch transactions: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch transactions")

# Keep all your existing content generation endpoints
@app.post("/api/generate", response_model=GenerateResponse)
async def generate_content(request: GenerateRequest):
    """Generate pure book content using AI providers"""
    try:
        if not request.prompt:
            raise HTTPException(status_code=400, detail="Prompt is required")
        
        content = ""
        used_provider = request.provider
        cover_image = None
        content_images = []
        
        # Generate text content first
        if request.provider in ["openai", "both"]:
            if not openai_client:
                logger.error("OpenAI client not initialized due to missing/invalid API key.")
                if request.provider != "both":
                    raise HTTPException(status_code=500, detail="OpenAI provider is unavailable.")
            else:
                try:
                    raw_content = await generate_with_openai(
                        openai_client,
                        request.prompt, 
                        request.max_tokens, 
                        request.temperature,
                        request.image_config
                    )
                    content = extract_pure_book_content(raw_content)
                    used_provider = "openai"
                    logger.info("Successfully generated content with OpenAI")
                except Exception as e:
                    logger.error(f"OpenAI generation failed: {str(e)}")
                    if request.provider != "both":
                        raise HTTPException(status_code=500, detail=f"OpenAI generation failed: {str(e)}")
        
        if (request.provider in ["gemini", "both"]) and not content:
            try:
                raw_content = await generate_with_gemini(request.prompt, request.image_config)
                content = extract_pure_book_content(raw_content)
                used_provider = "gemini"
                logger.info("Successfully generated content with Gemini")
            except Exception as e:
                logger.error(f"Gemini generation failed: {str(e)}")
                if not content:
                    raise HTTPException(status_code=500, detail=f"All AI providers failed. Last error: {str(e)}")
        
        if not content:
            raise HTTPException(status_code=500, detail="Failed to generate content with any provider")
        
        # Generate cover image if requested
        if request.generate_cover:
            try:
                cover_image = await generate_cover_image(request.prompt, request.cover_style, used_provider)
            except Exception as e:
                logger.error(f"Cover image generation failed: {str(e)}")
        
        # Generate content images if requested
        if request.image_config and request.image_config.include_images:
            try:
                content_images = await generate_content_images(
                    content, 
                    request.image_config, 
                    used_provider
                )
            except Exception as e:
                logger.error(f"Content image generation failed: {str(e)}")
        
        word_count = len(content.split())
        
        return GenerateResponse(
            content=content,
            provider=used_provider,
            timestamp=datetime.now().isoformat(),
            word_count=word_count,
            cover_image=cover_image,
            images=content_images
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in generate_content: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Keep all your existing helper functions
async def generate_with_openai(client: openai.OpenAI, prompt: str, max_tokens: int, temperature: float, image_config: Optional[ImageConfig] = None) -> str:
    """Generate content using OpenAI GPT-4 with strict book content focus"""
    try:
        image_instruction = ""
        if image_config and image_config.include_images:
            image_instruction = f"""

Include {image_config.image_count} strategic image placement suggestions using: [IMAGE: Brief description]
Place these markers where images would enhance understanding.
"""

        system_content = f"""You are a professional book writer. Write ONLY the book content - no commentary, explanations, or assistant responses.

CRITICAL RULES:
- Write ONLY the book content itself
- NO "Here's the chapter" or "I've created" or similar phrases
- NO explanations about what you're doing
- NO questions to the user
- NO suggestions or offers for help
- Start directly with the book content
- End with the book content
- Be a book, not an assistant

BOOK WRITING GUIDELINES:
- Clear, professional writing style
- Practical examples and actionable insights
- Proper headings and structure
- Substantial and valuable content
- Engaging and informative tone
- Commercial publication quality{image_instruction}

Write pure book content only. No assistant text whatsoever."""

        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": system_content
                    },
                    {
                        "role": "user",
                        "content": f"Write the book content for: {prompt}"
                    }
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                presence_penalty=0.1,
                frequency_penalty=0.1
            )
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"OpenAI API error: {str(e)}")
        raise e

async def generate_with_gemini(prompt: str, image_config: Optional[ImageConfig] = None) -> str:
    """Generate content using Google Gemini with strict book content focus"""
    try:
        image_instruction = ""
        if image_config and image_config.include_images:
            image_instruction = f"""

Include {image_config.image_count} strategic image placement suggestions using: [IMAGE: Brief description]
Place these markers where images would enhance understanding.
"""

        enhanced_prompt = f"""You are a professional book writer. Write ONLY the book content - no commentary, explanations, or assistant responses.

CRITICAL RULES:
- Write ONLY the book content itself
- NO "Here's the chapter" or "I've created" or similar phrases  
- NO explanations about what you're doing
- NO questions to the user
- NO suggestions or offers for help
- Start directly with the book content
- End with the book content
- Be a book, not an assistant

BOOK WRITING GUIDELINES:
- Clear, professional writing style
- Practical examples and actionable insights
- Proper headings and structure  
- Substantial and valuable content
- Engaging and informative tone
- Commercial publication quality{image_instruction}

Write pure book content for: {prompt}

Write only the book content. No assistant text whatsoever."""
        
        # Try different Gemini models in order of preference
        model_names = [
            'gemini-2.0-flash-exp',
            'gemini-1.5-flash', 
            'gemini-1.5-pro',
            'gemini-2.5-flash-exp',
        ]
        
        last_error = None
        
        for model_name in model_names:
            try:
                logger.info(f"Trying Gemini model: {model_name}")
                model = genai.GenerativeModel(model_name)
                
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: model.generate_content(enhanced_prompt)
                )
                
                logger.info(f"Successfully used Gemini model: {model_name}")
                return response.text.strip()
                
            except Exception as e:
                logger.warning(f"Failed with model {model_name}: {str(e)}")
                last_error = e
                continue
        
        raise last_error if last_error else Exception("No Gemini models available")
                
    except Exception as e:
        logger.error(f"Gemini API error: {str(e)}")
        raise e

async def generate_cover_image(book_prompt: str, style: str, provider: str) -> Optional[CoverImage]:
    """Generate a cover image for the book"""
    try:
        cover_prompt = f"Book cover design for: {book_prompt[:100]}... Style: {style}, professional book cover, title space, clean design, high quality"
        
        if provider == "openai" and openai_client:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: openai_client.images.generate(
                    model="dall-e-3",
                    prompt=cover_prompt,
                    size="1024x1024",
                    quality="standard",
                    n=1,
                )
            )
            
            return CoverImage(
                url=response.data[0].url,
                prompt_used=cover_prompt,
                style=style
            )
        else:
            unsplash_url = await get_unsplash_image(f"book cover {style}")
            if unsplash_url:
                return CoverImage(
                    url=unsplash_url,
                    prompt_used=f"Unsplash: book cover {style}",
                    style=style
                )
    except Exception as e:
        logger.error(f"Cover image generation failed: {str(e)}")
        return None

async def generate_content_images(content: str, image_config: ImageConfig, provider: str) -> List[ImageData]:
    """Generate images for book content based on image markers"""
    images = []
    
    try:
        import re
        image_markers = re.findall(r'\[IMAGE: ([^\]]+)\]', content)
        image_markers = image_markers[:image_config.image_count]
        
        for i, description in enumerate(image_markers):
            try:
                image_data = None
                
                if image_config.image_source in ["generate", "both"] and provider == "openai" and openai_client:
                    try:
                        response = await asyncio.get_event_loop().run_in_executor(
                            None,
                            lambda: openai_client.images.generate(
                                model="dall-e-3",
                                prompt=f"{description}, {image_config.image_style} style, high quality, detailed",
                                size="1024x1024",
                                quality="standard",
                                n=1,
                            )
                        )
                        
                        image_data = ImageData(
                            url=response.data[0].url,
                            caption=description,
                            source="generated",
                            alt_text=description
                        )
                    except Exception as e:
                        logger.error(f"DALL-E generation failed for image {i}: {str(e)}")
                
                if not image_data and image_config.image_source in ["unsplash", "both"]:
                    unsplash_url = await get_unsplash_image(description)
                    if unsplash_url:
                        image_data = ImageData(
                            url=unsplash_url,
                            caption=description,
                            source="unsplash",
                            alt_text=description
                        )
                
                if image_data:
                    images.append(image_data)
                    
            except Exception as e:
                logger.error(f"Failed to generate image for description '{description}': {str(e)}")
                continue
                
    except Exception as e:
        logger.error(f"Content image generation failed: {str(e)}")
    
    return images

async def get_unsplash_image(query: str) -> Optional[str]:
    """Get a copyright-free image from Unsplash"""
    try:
        unsplash_source_url = f"https://source.unsplash.com/1200x800/?{query.replace(' ', ',')}"
        
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: requests.head(unsplash_source_url, timeout=10)
        )
        
        if response.status_code == 200:
            return unsplash_source_url
        else:
            fallback_url = f"https://source.unsplash.com/1200x800/?abstract,minimal"
            return fallback_url
            
    except Exception as e:
        logger.error(f"Unsplash image fetch failed: {str(e)}")
        return None

@app.get("/api/providers")
async def get_available_providers():
    """Get list of available AI providers"""
    providers = []
    
    if openai_client and os.getenv("OPENAI_API_KEY"):
        providers.append({
            "name": "openai",
            "display_name": "OpenAI GPT-4 + DALL-E",
            "available": True,
            "features": ["text_generation", "image_generation"]
        })
    else:
        providers.append({
            "name": "openai",
            "display_name": "OpenAI GPT-4 + DALL-E",
            "available": False,
            "reason": "API key not configured",
            "features": ["text_generation", "image_generation"]
        })
    
    if os.getenv("GEMINI_API_KEY"):
        providers.append({
            "name": "gemini",
            "display_name": "Google Gemini (2.0/1.5 Flash)",
            "available": True,
            "features": ["text_generation"],
            "models_tried": ["gemini-2.0-flash-exp", "gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.5-flash-exp"]
        })
    else:
        providers.append({
            "name": "gemini",
            "display_name": "Google Gemini (2.0/1.5 Flash)",
            "available": False,
            "reason": "API key not configured",
            "features": ["text_generation"]
        })
    
    return {"providers": providers}

@app.get("/api/test-gemini")
async def test_gemini_models():
    """Test endpoint to check which Gemini models are available"""
    if not os.getenv("GEMINI_API_KEY"):
        return {"error": "GEMINI_API_KEY not configured"}
    
    results = {}
    test_prompt = "Write one paragraph about artificial intelligence in professional writing style."
    
    model_names = [
        'gemini-2.0-flash-exp',
        'gemini-1.5-flash',
        'gemini-1.5-pro', 
        'gemini-2.5-flash-exp'
    ]
    
    for model_name in model_names:
        try:
            model = genai.GenerativeModel(model_name)
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda m=model: m.generate_content(test_prompt)
            )
            results[model_name] = {
                "status": "success",
                "response": response.text.strip()[:100] + "..." if len(response.text) > 100 else response.text.strip()
            }
        except Exception as e:
            results[model_name] = {
                "status": "failed",
                "error": str(e)
            }
    
    return {"model_test_results": results}

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AI Book Generator API - Pure Content Focus + Payments", 
        "version": "2.2.0", 
        "docs": "/docs",
        "features": ["pure_book_content", "content_filtering", "cover_images", "content_images", "razorpay_payments", "wallet_system"],
        "improvements": ["No assistant commentary", "Pure book content extraction", "Advanced content filtering", "Secure payment processing"]
    }

# if __name__ == "__main__":
#     port = int(os.getenv("PORT", 8000))
    
#     print(f"""
#     🚀 AI Book Generator Backend v2.2 Starting...
    
#     📍 Server: http://localhost:{port}
#     📖 API Docs: http://localhost:{port}/docs
#     🔍 Health Check: http://localhost:{port}/api/health
    
#     ✨ NEW FEATURES v2.2:
#     - 🎯 Pure book content extraction
#     - 🧹 Advanced content filtering
#     - ❌ No assistant commentary
#     - 📚 Professional book output only
#     - 💳 Razorpay payment integration
#     - 💰 Wallet system
    
#     🔑 Environment Variables:
#     - OPENAI_API_KEY: {'✅ Set' if os.getenv('OPENAI_API_KEY') else '❌ Missing'}
#     - GEMINI_API_KEY: {'✅ Set' if os.getenv('GEMINI_API_KEY') else '❌ Missing'}
#     - RAZORPAY_KEY_ID: {'✅ Set' if os.getenv('RAZORPAY_KEY_ID') else '❌ Missing'}
#     - RAZORPAY_KEY_SECRET: {'✅ Set' if os.getenv('RAZORPAY_KEY_SECRET') else '❌ Missing'}
#     """)
    
#     uvicorn.run(
#         "main:app",
#         host="0.0.0.0",
#         port=port,
#         reload=True,
#         log_level="info"
#     )

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    
    # Determine if we're in production
    is_production = os.getenv("ENVIRONMENT", "development").lower() == "production"
    
    print(f"""
    🚀 AI Book Generator Backend v2.2 Starting...
    
    📍 Environment: {'Production' if is_production else 'Development'}
    📍 Server: {'0.0.0.0' if is_production else 'localhost'}:{port}
    📖 API Docs: http://{'0.0.0.0' if is_production else 'localhost'}:{port}/docs
    🔍 Health Check: http://{'0.0.0.0' if is_production else 'localhost'}:{port}/api/health
    
    ✨ NEW FEATURES v2.2:
    - 🎯 Pure book content extraction
    - 🧹 Advanced content filtering
    - ❌ No assistant commentary
    - 📚 Professional book output only
    - 💳 Razorpay payment integration
    - 💰 Wallet system
    
    🔑 Environment Variables:
    - OPENAI_API_KEY: {'✅ Set' if os.getenv('OPENAI_API_KEY') else '❌ Missing'}
    - GEMINI_API_KEY: {'✅ Set' if os.getenv('GEMINI_API_KEY') else '❌ Missing'}
    - RAZORPAY_KEY_ID: {'✅ Set' if os.getenv('RAZORPAY_KEY_ID') else '❌ Missing'}
    - RAZORPAY_KEY_SECRET: {'✅ Set' if os.getenv('RAZORPAY_KEY_SECRET') else '❌ Missing'}
    """)
    
    # Run the server
    uvicorn.run(
        "main:app",  # Important: Use string format for production
        host="0.0.0.0",  # Important: Listen on all interfaces
        port=port,
        reload=not is_production,  # Only reload in development
        log_level="info" if is_production else "debug",
        access_log=True
    )