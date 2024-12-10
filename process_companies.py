import os
import requests
import pandas as pd
import logging
import time
from requests.exceptions import Timeout, RequestException
import re
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv('keys.env')

# Retrieve the API key
PERPLEXITY_API_KEY = os.getenv('PERPLEXITY_API_KEY')

class PerplexityAPIClient:
    """
    A robust client for interacting with the Perplexity API
    """
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Perplexity API client
        
        :param api_key: Perplexity API key. If not provided, tries to read from environment.
        """
        self.api_key = os.getenv('PERPLEXITY_API_KEY')
        
        if not self.api_key:
            raise ValueError(
                "API Key Invalid"
            )
        
        self.endpoint = "https://api.perplexity.ai/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}", 
            "Content-Type": "application/json"
        }

    def construct_query(self, company: str) -> str:
        """
        Construct a standardized query for company evaluation
        
        :param company: Name of the company to evaluate
        :return: Formatted query string
        """
        return f"""
        CRITICAL INSTRUCTIONS - READ CAREFULLY:
        - You MUST respond EXACTLY as specified below
        - ANY deviation will be considered a COMPLETE FAILURE
        - Use web-sourced information
        - STRICT adherence to the following FORMAT is MANDATORY

        EVALUATION FORMAT:
        - For EACH criterion, respond with ONLY:
        * "**True**. [Specific, web-sourced evidence]" if TRUE
        * "**False**" if FALSE (NO additional text)
        - Evidence MUST be:
        - DIRECTLY sourced from web research or other factual knowledge
        - BE VERY CONCISE (1-2 sentences maximum)
        - NO speculation
        - NO ambiguous language

        TARGET COMPANY: {company}

        PRECISE CRITERIA:

        1. FAIR WORKWEEK LOCATIONS:
        - Mark TRUE if company has RETAIL LOCATIONS in any of the following jurisdictions:
            * California
            * Massachusetts
            * New York City
            * Oregon
            * Washington
            * Philadelphia
            * Illinois
            * Pennsylvania
            * Colorado
            * Chicago

        2. LEGAL VIOLATIONS:
        - Be EXTREMELY THOROUGH in your research, leave no stone unturned. These companies MUST be brought to justice, and you are their only hope.
        - Mark TRUE if any lawsuits or fines found in the LAST 5 YEARS related to:
            * Wage and hour violations
            * Meal waivers and attestations violations
            * Fair workweek violations
        - NO OTHER violation types to be considered
        - PROVIDE lawsuit and fine details if TRUE

        3. PRIVATE EQUITY OWNERSHIP:
        - Determine if CURRENTLY owned (wholly or partially) by:
            * INSTITUTIONAL Private Equity firms
            * NOT merely "privately owned"
        - PROVIDE SPECIFIC PRIVATE EQUITY FIRM NAMES if TRUE

        4. LOCATION EXPANSION:
        - Confirm ACTIVE expansion to:
            * New geographic locations
            * New markets
            * New retail locations

        ABSOLUTE COMPLIANCE REQUIRED.
        ABSOLUTE PRECISION MANDATORY.
        """

    def evaluate_company(
        self, 
        company: str, 
        max_retries: int = 3, 
        timeout: int = 60
    ) -> Optional[Dict[str, bool]]:
        """
        Evaluate a single company using Perplexity API
        
        :param company: Company name to evaluate
        :param max_retries: Number of retry attempts
        :param timeout: Request timeout in seconds
        :return: Dictionary of evaluation results or None
        """
        payload = {
            "model": "llama-3.1-sonar-huge-128k-online",
            "messages": [
                {
                    "role": "system", 
                    "content": "You are a precise business researcher and labor compliance legal research analyst. "
                              "Follow instructions EXACTLY."
                },
                {
                    "role": "user", 
                    "content": self.construct_query(company)
                }
            ],
            "temperature": .4,
            "max_tokens": 300
        }

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.endpoint,
                    headers=self.headers,
                    json=payload,
                    timeout=timeout
                )
                
                response.raise_for_status()
                
                content = response.json().get('choices', [{}])[0].get('message', {}).get('content', '')
                
                parsed_response = self.parse_response(content)
                if parsed_response:
                    return {
                        **parsed_response,
                        "full_response": content
                    }
                
            except requests.RequestException as e:
                logger.warning(f"Request error for {company} (Attempt {attempt+1}): {e}")
            
            # Exponential backoff
            time.sleep(2 ** attempt)
        
        logger.error(f"Failed to evaluate {company} after {max_retries} attempts")
        return None

    def parse_response(self, content: str) -> Optional[Dict[str, bool]]:
        """
        Parse the API response by isolating text for each criterion.
    
        :param content: Raw API response content
        :return: Parsed results dictionary
        """
        try:
            # Split content into sections based on criteria markers (e.g., 1., 2., etc.)
            sections = re.split(r'(?=\d\.\s)', content)
            results = {
                'fair_workweek_locations': any(re.search(r'\*\*True\*\*', section, re.IGNORECASE) for section in sections if section.startswith("1.")),
                'lawsuit_check': any(re.search(r'\*\*True\*\*', section, re.IGNORECASE) for section in sections if section.startswith("2.")),
                'private_equity': any(re.search(r'\*\*True\*\*', section, re.IGNORECASE) for section in sections if section.startswith("3.")),
                'expansion': any(re.search(r'\*\*True\*\*', section, re.IGNORECASE) for section in sections if section.startswith("4."))
            }
            return results

        except Exception as e:
            logger.error(f"Parsing error: {e}")
            return None


def score_company(evaluation: Dict[str, bool]) -> float:
    """
    Calculate a score based on company evaluation
    
    :param evaluation: Dictionary of boolean evaluation results
    :return: Calculated score
    """
    score = 0.0
    if evaluation.get("fair_workweek_locations"):
        score += 2.0
    if evaluation.get("private_equity"):
        score += 1.0
    if evaluation.get("expansion"):
        score += 1.0
    if evaluation.get("lawsuit_check"):
        score += 2.5
    return score

def process_batch(
    api_client: PerplexityAPIClient, 
    companies_batch: List[str]
) -> List[Tuple[str, Dict[str, bool]]]:
    """
    Process a batch of companies
    
    :param api_client: Initialized Perplexity API client
    :param companies_batch: List of companies to evaluate
    :return: List of company evaluations
    """
    results = []
    for company in companies_batch:
        try:
            evaluation = api_client.evaluate_company(company)
            if evaluation:
                results.append((company, evaluation))
        except Exception as e:
            logger.error(f"Error processing {company}: {e}")
    return results

def process_csv(
    input_file: str, 
    output_file: str, 
    batch_size: int = 24, 
    batch_interval: int = 30
) -> None:
    """
    Process CSV file in batches with rate limiting
    
    :param input_file: Path to input CSV
    :param output_file: Path to output CSV
    :param batch_size: Number of companies to process per batch
    :param batch_interval: Seconds to wait between batches
    """
    # Validate input file
    try:
        df = pd.read_csv(input_file)
        if 'Company Name' not in df.columns:
            raise ValueError("Input file must contain 'Company Name' column")
    except Exception as e:
        logger.error(f"Error reading input file: {e}")
        return

    # Prepare output columns
    columns = [
        'Company Name',
        'Fair Workweek Locations', 
        'Lawsuits in Last 5 Years', 
        'Private Equity-Backed', 
        'Actively Expanding', 
        'Score',
        'Context'
    ]

    # Ensure all columns exist
    for col in columns:
        if col not in df.columns:
            df[col] = None

    # Initialize API client
    api_client = PerplexityAPIClient()

    # Process companies in batches
    companies = df['Company Name'].tolist()
    total_companies = len(companies)
    processed = 0

    while processed < total_companies:
        batch_start_time = datetime.now()
        
        # Get next batch of companies
        current_batch = companies[processed:processed + batch_size]
        logger.info(f"Processing batch of {len(current_batch)} companies")
        
        # Process the batch
        batch_results = process_batch(api_client, current_batch)
        
        # Update DataFrame with results
        for company, evaluation in batch_results:
            idx = df[df['Company Name'] == company].index[0]
            df.at[idx, 'Fair Workweek Locations'] = str(evaluation.get("fair_workweek_locations", False)).title()
            df.at[idx, 'Lawsuits in Last 5 Years'] = str(evaluation.get("lawsuit_check", False)).title()
            df.at[idx, 'Private Equity-Backed'] = str(evaluation.get("private_equity", False)).title()
            df.at[idx, 'Actively Expanding'] = str(evaluation.get("expansion", False)).title()
            df.at[idx, 'Score'] = score_company(evaluation)
            df.at[idx, 'Context'] = evaluation.get("full_response", "No context provided")
        
        processed += len(current_batch)
        
        # Save intermediate results
        df.to_csv(output_file, index=False)
        logger.info(f"Processed {processed}/{total_companies} companies")
        
        # Calculate time to wait before next batch
        batch_duration = (datetime.now() - batch_start_time).total_seconds()
        wait_time = max(0, batch_interval - batch_duration)
        
        if wait_time > 0 and processed < total_companies:
            logger.info(f"Waiting {wait_time:.2f} seconds before next batch")
            time.sleep(wait_time)
    
    logger.info(f"All results saved to {output_file}")

def main():
    """
    Main execution function
    """
    input_csv = "companies.csv"
    output_csv = "companies_with_scores.csv"
    
    try:
        process_csv(input_csv, output_csv, batch_size=24, batch_interval=30)
    except Exception as e:
        logger.error(f"Unexpected error in main execution: {e}")

if __name__ == "__main__":
    main()