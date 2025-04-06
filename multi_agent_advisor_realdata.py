import google.generativeai as genai
import os
import time
import re # Regular expressions for parsing data requests
import yfinance as yf
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv()

try:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY environment variable not set.")
    genai.configure(api_key=GOOGLE_API_KEY)
    print("Google AI SDK configured successfully.")
except Exception as e:
    print(f"Error configuring Google AI SDK: {e}")
    exit()

# --- Model Selection and Configuration ---
MODEL_NAME = "gemini-2.5-pro-exp-03-25" # Or gemini-pro / gemini-1.5-pro-latest
generation_config = {
    "temperature": 0.5, # Slightly lower for more focused financial output
    "top_p": 1.0,
    "top_k": 32,
    "max_output_tokens": 50000,
}
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

try:
    model = genai.GenerativeModel(
        model_name=MODEL_NAME,
        generation_config=generation_config,
        safety_settings=safety_settings
    )
    print(f"Using model: {MODEL_NAME}")
except Exception as e:
    print(f"Error initializing generative model: {e}")
    exit()

# --- Tool: Financial Data Fetcher ---

def get_market_data(tickers):
    """
    Fetches basic market data for a list of tickers using yfinance.
    Args:
        tickers (list): A list of stock/index tickers (e.g., ['AAPL', '^GSPC']).
    Returns:
        dict: A dictionary where keys are tickers and values are strings
              containing fetched data or error messages. Returns None if yfinance fails.
    """
    if not tickers:
        return {}
    print(f"--- Fetching market data for: {', '.join(tickers)} ---")
    data = {}
    not_found = []
    try:
        for ticker_symbol in tickers:
            ticker_obj = yf.Ticker(ticker_symbol)
            info = ticker_obj.info # Basic info dictionary

            # Check if data was found - yfinance sometimes returns minimal dict for invalid tickers
            if info and info.get('regularMarketPrice') is not None:
                # Extract key info - adjust as needed
                price = info.get('regularMarketPrice', 'N/A')
                prev_close = info.get('previousClose', 'N/A')
                volume = info.get('regularMarketVolume', 'N/A')
                day_high = info.get('dayHigh', 'N/A')
                day_low = info.get('dayLow', 'N/A')
                fifty_two_week_high = info.get('fiftyTwoWeekHigh', 'N/A')
                fifty_two_week_low = info.get('fiftyTwoWeekLow', 'N/A')
                name = info.get('shortName', ticker_symbol) # Use shortName if available

                data[ticker_symbol] = (
                    f"Data for {name} ({ticker_symbol}):\n"
                    f"  Current Price: ~${price}\n"
                    f"  Previous Close: ${prev_close}\n"
                    f"  Day Range: ${day_low} - ${day_high}\n"
                    f"  52 Week Range: ${fifty_two_week_low} - ${fifty_two_week_high}\n"
                    f"  Volume: {volume:,}" # Format volume with commas
                 )
            elif info and info.get('marketCap') is None and info.get('regularMarketPrice') is None:
                 # Heuristic: If it lacks market cap AND price, likely not found well
                 print(f"Warning: Potentially incomplete data for {ticker_symbol}. Treating as not found.")
                 not_found.append(ticker_symbol)
                 data[ticker_symbol] = f"Error: Could not retrieve detailed data for {ticker_symbol}."
            else:
                # Sometimes yfinance returns *some* info but not critical price, handle this
                print(f"Warning: Missing key price data for {ticker_symbol}, attempting fallback info.")
                name = info.get('shortName', ticker_symbol)
                currency = info.get('currency', '?')
                market_state = info.get('marketState', 'Unknown')
                market_cap = info.get('marketCap')
                mc_str = f"${market_cap:,}" if market_cap else "N/A"

                data[ticker_symbol] = (
                     f"Limited Data for {name} ({ticker_symbol}):\n"
                     f"  Currency: {currency}\n"
                     f"  Market State: {market_state}\n"
                     f"  Market Cap: {mc_str}"
                )
                if not info.get('regularMarketPrice'): # If price truly missing
                     data[ticker_symbol] += "\n  *Current Price data unavailable via this tool.*"


        # Handle explicitly not found tickers after loop
        for nf_ticker in not_found:
             if nf_ticker not in data: # Ensure we didn't overwrite somehow
                data[nf_ticker] = f"Error: Ticker symbol '{nf_ticker}' not found or data unavailable."

        return data

    except Exception as e:
        print(f"!!! Error fetching data using yfinance: {e}")
        # Return errors for all requested tickers if the API call failed globally
        error_data = {ticker: f"Error fetching data: {e}" for ticker in tickers}
        return error_data


def extract_ticker_requests(text):
    """
    Simple extraction of potential stock tickers (e.g., AAPL, GOOGL, ^GSPC)
    from text using regular expressions. This is basic and may need refinement.
    """
    # Matches 1-6 uppercase letters/digits, optionally preceded by '^' (for indices)
    # Uses word boundaries (\b) to avoid matching parts of words.
    tickers = re.findall(r'\b([A-Z0-9\.]{1,6})\b', text)
    index_tickers = re.findall(r'\b(\^[A-Z0-9]{1,6})\b', text) # Specifically find indices like ^GSPC

    # Combine and deduplicate, preferring uppercase
    all_tickers = set([t.upper() for t in tickers + index_tickers])

    # Basic filtering: remove common words that might look like tickers
    common_words_to_filter = {'A', 'I', 'IS', 'IT', 'DO', 'BE', 'ALL', 'FOR', 'NOW', 'THE', 'AND', 'ARE'}
    filtered_tickers = [t for t in all_tickers if t not in common_words_to_filter]

    if filtered_tickers:
         print(f"--- Detected potential ticker symbols for data fetching: {filtered_tickers} ---")
    return filtered_tickers

# --- Agent Definitions (Modified Market Analyst Persona) ---

class FinancialAgent:
    """Represents an AI agent with a specific financial role."""
    def __init__(self, role, persona, model):
        self.role = role
        self.persona = persona
        self.model = model
        self.conversation_history = [] # Store context for potential follow-ups within a session

    def process(self, input_prompt, is_follow_up=False):
        """
        Processes the prompt using the agent's model.
        Handles conversation history within the agent's turn.
        """
        full_prompt = f"{self.persona}\n\n"
        if self.conversation_history:
             full_prompt += "Previous Conversation Turn Context:\n" + "\n".join(self.conversation_history) + "\n\n"
        full_prompt += f"Current Task/Input:\n{input_prompt}"

        print(f"\n--- Sending task to {self.role}... ---")

        try:
            response = self.model.generate_content(full_prompt)

            if not response.candidates or not response.candidates[0].content.parts:
                 print(f"Warning: {self.role} received an empty or invalid response.")
                 try:
                    # Attempt fallback access
                    if hasattr(response, 'text') and response.text:
                        result_text = response.text
                        print(f"--- {self.role} finished (using response.text fallback). ---")
                    else:
                         raise ValueError("Response candidate empty and no text fallback")
                 except Exception as fallback_e:
                     print(f"Error during fallback access for {self.role}: {fallback_e}")
                     # Include feedback if available
                     feedback = getattr(response, 'prompt_feedback', 'No feedback available')
                     return f"[{self.role} Error: Received no valid content. Feedback: {feedback}]"
            else:
                result_text = response.text # Access text safely

            print(f"--- {self.role} finished. ---")

            # Add exchange to history for this agent *if* it's part of an ongoing logical turn
            # For this structure, we might not need complex history per agent yet.
            # self.conversation_history.append(f"Input: {input_prompt}")
            # self.conversation_history.append(f"Output: {result_text}")
            # Limit history size if needed

            return result_text

        except Exception as e:
            print(f"!!! Error during generation for {self.role}: {e}")
            try:
                if response and response.prompt_feedback:
                    print(f"Prompt Feedback: {response.prompt_feedback}")
            except AttributeError:
                pass
            return f"[{self.role} Error: Generation failed - {e}]"

    def reset_history(self):
        self.conversation_history = []

# --- Orchestration (Modified Flow) ---

class AdvisorTeam:
    """Manages the team of financial agents and the consultation process."""
    def __init__(self, model):
        self.model = model
        self.agents = self._create_agents()

    def _create_agents(self):
        """Creates instances of all financial agents."""
        # Agents Dictionary - Define Roles and Personas Here
        agents = {
            "Market Analyst": FinancialAgent(
                role="Market Analyst",
                persona="""You are an expert Market Analyst. Your focus is on current market conditions, economic indicators (inflation, interest rates, GDP growth), sector trends, and geopolitical events relevant to the user's query.
                **Your First Task:** Analyze the user's query and the broader economic landscape. Identify if specific, current market data (like stock prices for mentioned companies/indices e.g., AAPL, TSLA, ^GSPC for S&P 500, ^IXIC for Nasdaq, ^DJI for Dow Jones) is needed for a deeper analysis. If so, CLEARLY LIST the ticker symbols you require, like: 'Data needed for: AAPL, ^GSPC'.
                **Your Second Task (if data is provided):** Incorporate the provided real-time market data into your analysis, discussing recent performance, volatility (if inferrable from data), and how it relates to the overall market context and the user's query. Provide data-driven insights.
                Do not give investment recommendations directly. Focus on objective market analysis.""",
                model=self.model
            ),
            "Risk Assessor": FinancialAgent(
                role="Risk Assessor",
                persona="""You are a cautious Risk Assessor. Your job is to identify potential risks, downsides, volatility, and suitability based on general risk tolerance concepts (e.g., conservative, moderate, aggressive). Analyze the market context (including any fetched data provided) and the user's query specifics to highlight potential dangers or challenges associated with the topic discussed. Consider both market risks and any potential strategy risks. Do *not* give specific investment advice.""",
                model=self.model
            ),
            "Planning Strategist": FinancialAgent(
                role="Planning Strategist",
                 persona="""You are a forward-thinking Planning Strategist. Considering the market analysis (including any fetched data) and the risk assessment, you outline potential strategies and approaches relevant to the user's query. Discuss concepts like asset allocation (in general terms), diversification, time horizons, and potentially relevant *types* of financial products or accounts (e.g., ETFs, mutual funds, retirement accounts like 401k/IRA). Do NOT recommend specific stocks, funds, or make personalized suitability judgments. Focus on strategic concepts and possibilities.""",
                model=self.model
            ),
            "Chief Advisor (Synthesizer)": FinancialAgent(
                role="Chief Advisor (Synthesizer)",
                persona="""You are the Chief Advisor leading this consultation. Your role is to synthesize the insights from the Market Analyst, Risk Assessor, and Planning Strategist into a single, coherent, balanced, and professional-sounding final response for the user. Address the user directly and adopt a helpful, informative, yet cautious tone appropriate for financial discussion.
                Ensure the final output integrates the key points, including any fetched data insights, from all prior analyses. It should read like a unified recommendation from a team.
                **Crucially, conclude your response with the following mandatory disclaimer, verbatim:**

                ---
                **IMPORTANT DISCLAIMER:** This information is generated by an AI model using public data and predictive algorithms. It is intended for informational purposes ONLY and does **NOT** constitute personalized financial advice, solicitation, or endorsement of any particular investment strategy or product. Financial markets are complex and volatile; past performance is not indicative of future results. AI models may have limitations, inaccuracies, or biases in their data or analysis. **Before making any financial decisions, you MUST consult with a qualified, licensed human financial advisor or planner who can assess your individual circumstances, risk tolerance, and financial goals.** Relying solely on this AI output for financial decisions could lead to significant losses.
                ---""",
                model=self.model
            )
        }
        # Add other potential agents: e.g., Tax Implications Analyst, Behavioral Finance Coach etc.
        return agents

    def run_consultation(self, user_query):
        """Orchestrates the interaction, including data fetching."""
        print("\nInitiating financial consultation protocol...")
        start_time = time.time()

        # Reset agent history if needed (for stateless operation per query)
        for agent in self.agents.values():
            agent.reset_history()

        # 1. Initial Market Analysis + Data Request Identification
        market_analyst = self.agents["Market Analyst"]
        initial_analysis_prompt = f"User Query: \"{user_query}\"\n\nAnalyze this query and the general market. Identify specific ticker symbols for current data if needed for a better analysis. List them clearly if required."
        initial_analysis_response = market_analyst.process(initial_analysis_prompt)

        if not initial_analysis_response or "[Market Analyst Error:" in initial_analysis_response:
            print("Consultation stopped due to error in initial Market Analysis.")
            return f"Error during consultation: {initial_analysis_response}"

        # 2. Data Fetching (if requested)
        requested_tickers = extract_ticker_requests(initial_analysis_response)
        fetched_data = {}
        fetched_data_summary = "No specific market data requested or fetched for this query."
        if requested_tickers:
            fetched_data = get_market_data(requested_tickers)
            if fetched_data:
                # Prepare a string summary of the fetched data for the next prompts
                fetched_data_summary = "Fetched Market Data:\n"
                for ticker, data_str in fetched_data.items():
                    fetched_data_summary += f"- {ticker}: {data_str}\n"
                fetched_data_summary += "---\n" # Separator
            else:
                 fetched_data_summary = "Attempted to fetch data, but failed.\n---\n"


        # 3. Refined Market Analysis (with data, if fetched)
        refined_analysis_prompt = f"""User Query: "{user_query}"
        Your Initial Thoughts:
        {initial_analysis_response}
        ---
        {fetched_data_summary}
        ---
        Now, refine your market analysis, incorporating the provided fetched data (if any). Focus on data-driven insights related to the user query.
        """
        # Call the Market Analyst again with the data context
        market_analysis = market_analyst.process(refined_analysis_prompt, is_follow_up=True)
        if not market_analysis or "[Market Analyst Error:" in market_analysis :
             print("Consultation stopped due to error in refined Market Analysis.")
             # Use initial analysis as fallback maybe? Or return error.
             return f"Error during consultation: {market_analysis}"


        # 4. Risk Assessment
        risk_context = f"""Market Analysis Provided (incorporating fetched data if any):
        {market_analysis}
        ---
        Original User Query: "{user_query}"
        """
        risk_assessment = self.agents["Risk Assessor"].process(
            input_prompt=f"Based on the market analysis and user query above, identify key risks." ,
            is_follow_up=False # Reset history context for this agent's turn
        )
        if not risk_assessment or "[Risk Assessor Error:" in risk_assessment :
             print("Consultation stopped due to error in Risk Assessment.")
             return f"Error during consultation: {risk_assessment}"

        # 5. Planning Strategy
        planning_context = f"""Market Analysis:
        {market_analysis}
        ---
        Risk Assessment:
        {risk_assessment}
        ---
        Original User Query: "{user_query}"
        """
        planning_strategy = self.agents["Planning Strategist"].process(
             input_prompt=f"Considering the analysis and risks above, outline potential strategies relevant to the user query.",
             is_follow_up=False
        )
        if not planning_strategy or "[Planning Strategist Error:" in planning_strategy :
             print("Consultation stopped due to error in Planning Strategy.")
             return f"Error during consultation: {planning_strategy}"

        # 6. Synthesize Final Response
        final_context = f"""Synthesize the following inputs into a coherent final response for the user, addressing their original query: "{user_query}". Integrate all points smoothly and conclude *exactly* with the mandatory disclaimer.

        --- START OF TEAM INPUTS ---

        [Market Analyst Input - Context & Data Analysis]:
        {market_analysis}

        [Risk Assessor Input - Potential Downsides]:
        {risk_assessment}

        [Planning Strategist Input - Strategic Approaches]:
        {planning_strategy}

        --- END OF TEAM INPUTS ---

        Remember to address the user directly and maintain a professional tone. Do NOT add any preamble before your response or any text after the mandatory disclaimer.
        """
        final_response = self.agents["Chief Advisor (Synthesizer)"].process(
            input_prompt=final_context, # Give synthesizer specific task
            is_follow_up=False
        )

        end_time = time.time()
        print(f"\nConsultation finished in {end_time - start_time:.2f} seconds.")

        # Just return the final synthesized response for the user
        return final_response if final_response else "[Chief Advisor Error: Synthesis failed.]"

# --- Main Execution ---

if __name__ == "__main__":
    if not model:
        print("Model not initialized. Exiting.")
        exit()

    advisor_team = AdvisorTeam(model)

    print("\nWelcome to the AI Financial Advisory Team Simulation (v2 - Data Enhanced)")
    print("Type your financial query below. Type 'quit' to exit.")
    print("---")
    print("**Reminder:** This tool provides AI-generated information for discussion purposes only.")
    print("It is NOT a substitute for professional financial advice. Always consult a qualified human advisor.")
    print("---")


    while True:
        user_query = input("\nYour financial question: ")
        if user_query.lower() == 'quit':
            break
        if not user_query.strip():
            print("Please enter a query.")
            continue

        # Run the enhanced consultation process
        final_advice = advisor_team.run_consultation(user_query)

        print("\n\n--- AI Advisory Team Response ---")
        print(final_advice)
        print("---------------------------------")


    print("\nExiting AI Financial Advisor Simulation.")