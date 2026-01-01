import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY not found in .env file")

# --- 1. SETUP THE BRAIN ---
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0
)

# --- 2. DEFINE MOHITH'S REAL DATA ---
# This is the "Context" the AI will read to answer questions.
resume_context = """
OFFICIAL PROFILE:
Name: MohithSai Malla
Role: Full-Stack Web Developer & DSA Lead
Contact: +91-9391036388 | mallamohith20@gmail.com
Location: Anakapalli, Andhra Pradesh, India
Links:
- LinkedIn: https://www.linkedin.com/in/mohithmalla/
- GitHub: https://github.com/MohithMalla
- Portfolio: https://mallamohith.netlify.app/

EDUCATION:
- B.Tech in Computer Science, Vignan's Institute of Information Technology (VIIT).
- CGPA: 8.94 (Current)

TECHNICAL SKILLS:
- Languages: Python, JavaScript (ES6+), Java, SQL, C.
- Frontend: React.js, Next.js, Tailwind CSS, Bootstrap, HTML5, CSS3.
- Backend: Node.js, Express.js, MongoDB.
- Tools: Git, GitHub, Postman, VS Code, Vercel, Netlify, Figma.
- Core Concepts: Data Structures & Algorithms (DSA), OOPs, REST APIs, Agile.

EXPERIENCE:
1. Web Developer (Official VIIT College Website) | Feb 2025 - Apr 2025
   - Led the redesign of the official college website using Next.js + Tailwind.
   - Achieved a 95+ Lighthouse performance score.
   - Implemented CI/CD pipelines via GitHub Actions.

2. Frontend Developer Intern (Safe Your Web) | 2023 - 2025
   - Built responsive UI modules with React.js + Tailwind.
   - Reduced page latency by 30% using lazy loading.
   - Integrated Redux Toolkit and JWT-based authentication.

KEY PROJECTS:
1. Service Marketplace (MERN Stack): Real-time service discovery app handling 50+ queries/min. Uses Google Maps API and Socket.io.
2. Easy Order (Smart Restaurant App): QR-based ordering system reducing wait times by 35%.
3. EcoTrack (Plastic Waste Tracker): Native React app for tracking recycling, uses Geolocation.
4. The Reverse Recruiter (Agentic AI): An AI agent that interviews recruiters (Portfolio Project).

ACHIEVEMENTS:
- Winner: Pixel Perfect Challenge 2025 (Best React.js website).
- Finalist: Smart India Hackathon (SIH 2024 & 2025) internal rounds.
- CodeChef: Max Rating 1470 (Top 10% in 60+ contests).
- LeetCode: Peak Rating 1672 (Solved 250+ problems).

CERTIFICATIONS & ROLES:
- DSA Lead at AlgoZenith Club.
- Certified in DSA with Java.
"""

# --- 3. DEFINE THE TOOL ---
@tool
def get_portfolio_info(query: str):
    """
    Call this tool when the user asks ANY question about MohithSai Malla.
    It contains his Resume, Skills, Projects, Contact info, and Experience.
    """
    return resume_context

tools = [get_portfolio_info]

# --- 4. CREATE THE AGENT ---
agent_executor = create_react_agent(llm, tools)

def run_chat(user_input: str):
    try:
        # We add a specific instruction to make the bot behave professionally
        system_instruction = (
            "You are the AI Assistant for MohithSai Malla's portfolio. "
            "Answer questions strictly based on the provided context. "
            "If asked for contact info, provide the email and LinkedIn. "
            "Be enthusiastic and professional."
        )
        
        # Combine instructions with user input
        full_prompt = f"{system_instruction}\n\nUser Question: {user_input}"
        
        response = agent_executor.invoke({"messages": [("user", full_prompt)]})
        
        # Extract and clean the content
        ai_message = response["messages"][-1]
        content = ai_message.content

        # Handle formatting (Gemini sometimes returns lists/dicts)
        if isinstance(content, list):
            final_text = ""
            for part in content:
                if isinstance(part, dict) and "text" in part:
                    final_text += part["text"]
                elif isinstance(part, str):
                    final_text += part
            return final_text
        elif isinstance(content, dict) and "text" in content:
            return content["text"]
        
        return str(content)
        
    except Exception as e:
        print(f"Server Error: {e}")
        return "I encountered an error. Please try again."