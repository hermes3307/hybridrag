import os
import argparse
import json
import sys
import time
from typing import List, Dict, Any, Optional, Tuple
import anthropic
import requests
import traceback
import logging

# Import the vector system components from your existing code
from paste import ManualVectorDB, ManualVectorizer, log_memory_usage, get_memory_usage

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ManualSearchRAG:
    def __init__(self, model_type: str = "claude", vector_db_dir: str = "manual_vector_db"):
        """
        Initialize the Manual Search RAG system
        
        Args:
            model_type (str): Model type to use (default: "claude")
            vector_db_dir (str): Path to vector database directory
        """
        self.model_type = model_type.lower()
        self.client = None
        self.vector_db = None
        self.vectorizer = None
        self.vector_db_dir = vector_db_dir
        
        # Claude API setup
        if self.model_type == "claude":
            logger.info(f"Initializing LLM model: {self.model_type}")
            
            # Get API key
            self.api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not self.api_key:
                raise ValueError("Environment variable 'ANTHROPIC_API_KEY' is not set.")
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def load_model(self):
        """Load the Claude API client"""
        try:
            if self.model_type == "claude":
                logger.info("Setting up Claude API client...")
                self.client = anthropic.Anthropic(api_key=self.api_key)
                logger.info("Claude API client setup complete!")
                
                # Available Claude models
                models = [
                    "claude-3-5-sonnet-20240620", 
                    "claude-3-opus-20240229", 
                    "claude-3-sonnet-20240229", 
                    "claude-3-haiku-20240307"
                ]
                logger.info("Available Claude models:")
                for i, model in enumerate(models, 1):
                    logger.info(f"{i}. {model}")
                
                return True
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
            
        except Exception as e:
            logger.error(f"Error setting up API client: {str(e)}")
            return False
    
    def load_vector_db(self, vector_db_dir: str = None):
        """Load the vector database and vectorizer"""
        if vector_db_dir:
            self.vector_db_dir = vector_db_dir
            
        logger.info(f"Loading vector database from: {self.vector_db_dir}")
        log_memory_usage("Vector DB loading start")
        
        try:
            # Initialize vectorizer
            self.vectorizer = ManualVectorizer()
            
            # Load vector database
            self.vector_db = ManualVectorDB.load(self.vector_db_dir, self.vectorizer.vector_size)
            
            logger.info(f"Vector database loaded successfully!")
            logger.info(f"- Total vectors: {self.vector_db.total_vectors:,}")
            logger.info(f"- Total chunks: {len(self.vector_db.chunks):,}")
            logger.info(f"- Available manuals: {len(self.vector_db.chunk_metadata.get('manual_titles', []))}")
            
            log_memory_usage("Vector DB loading complete")
            return True
            
        except Exception as e:
            logger.error(f"Error loading vector database: {str(e)}")
            return False
    
    def search_manual_solutions(self, 
                              problem_description: str,
                              programming_language: str = "",
                              technology_stack: str = "",
                              k: int = 10,
                              filter_manual: str = None,
                              filter_section_type: str = None) -> List[Dict[str, Any]]:
        """
        Search for relevant manual sections based on problem description
        
        Args:
            problem_description: Description of the problem to solve
            programming_language: Target programming language
            technology_stack: Technology stack context
            k: Number of results to return
            filter_manual: Filter by specific manual
            filter_section_type: Filter by section type (api, example, configuration, etc.)
            
        Returns:
            List[Dict]: Relevant manual sections with metadata
        """
        if not self.vector_db or not self.vectorizer:
            logger.error("Vector database not loaded")
            return []
        
        # Construct enhanced search query
        search_query_parts = [problem_description]
        
        if programming_language:
            search_query_parts.append(f"programming language: {programming_language}")
        
        if technology_stack:
            search_query_parts.append(f"technology: {technology_stack}")
        
        # Add technical keywords to improve search
        technical_keywords = [
            "implementation", "example", "code", "solution", "method", 
            "function", "API", "configuration", "setup", "tutorial"
        ]
        
        enhanced_query = " ".join(search_query_parts)
        
        logger.info(f"Searching for solutions: '{enhanced_query}'")
        logger.info(f"Search parameters: k={k}, filter_manual={filter_manual}, filter_section_type={filter_section_type}")
        
        try:
            # Search the vector database
            results = self.vector_db.search(
                query=enhanced_query,
                k=k,
                vectorizer=self.vectorizer,
                filter_manual=filter_manual,
                filter_section_type=filter_section_type
            )
            
            # Enhance results with additional context
            enhanced_results = []
            for result in results:
                chunk = result['chunk']
                metadata = chunk.get('metadata', {})
                
                # Calculate relevance score based on multiple factors
                relevance_score = result['similarity']
                
                # Boost score for certain section types
                section_type = metadata.get('section_type', '')
                if section_type in ['api', 'example', 'configuration']:
                    relevance_score *= 1.2
                elif section_type == 'error':
                    relevance_score *= 1.1
                
                # Boost score if programming language matches
                if programming_language and programming_language.lower() in chunk['text'].lower():
                    relevance_score *= 1.3
                
                enhanced_result = {
                    'rank': result['rank'],
                    'similarity': result['similarity'],
                    'relevance_score': min(relevance_score, 1.0),  # Cap at 1.0
                    'chunk': chunk,
                    'manual_title': chunk.get('manual_title', ''),
                    'section_key': chunk.get('section_key', ''),
                    'section_type': section_type,
                    'chapter': metadata.get('chapter', ''),
                    'text_length': chunk.get('text_length', len(chunk['text'])),
                    'context_summary': self._create_context_summary(chunk)
                }
                
                enhanced_results.append(enhanced_result)
            
            # Re-sort by relevance score
            enhanced_results.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            # Update ranks
            for i, result in enumerate(enhanced_results):
                result['rank'] = i + 1
            
            logger.info(f"Found {len(enhanced_results)} relevant manual sections")
            return enhanced_results
            
        except Exception as e:
            logger.error(f"Error during manual search: {str(e)}")
            return []
    
    def _create_context_summary(self, chunk: Dict) -> str:
        """Create a brief context summary for a chunk"""
        text = chunk.get('text', '')
        metadata = chunk.get('metadata', {})
        
        # Extract key information
        summary_parts = []
        
        if metadata.get('chapter'):
            summary_parts.append(f"Chapter: {metadata['chapter']}")
        
        if metadata.get('section_type'):
            summary_parts.append(f"Type: {metadata['section_type']}")
        
        # Get first few lines as preview
        lines = text.split('\n')[:3]
        preview = ' '.join(lines).strip()
        if len(preview) > 200:
            preview = preview[:200] + "..."
        
        summary_parts.append(f"Preview: {preview}")
        
        return " | ".join(summary_parts)
    
    def generate_solution_prompt(self, 
                               problem_description: str,
                               programming_language: str = "",
                               technology_stack: str = "",
                               manual_results: List[Dict] = None,
                               max_manual_refs: int = 5,
                               model_id: str = "claude-3-5-sonnet-20240620") -> Dict[str, str]:
        """
        Generate a prompt for Claude API to create a code solution
        
        Args:
            problem_description: The problem to solve
            programming_language: Target programming language
            technology_stack: Technology stack context
            manual_results: Relevant manual sections
            max_manual_refs: Maximum number of manual references to include
            model_id: Claude model to use
            
        Returns:
            Dict: Prompt information for Claude API
        """
        # System prompt
        system_prompt = """You are an expert software developer and technical writer who specializes in creating code solutions based on technical documentation and manuals.

Your task is to:
1. Analyze the provided problem description and technical requirements
2. Review the relevant manual sections and documentation provided
3. Create a complete, working code solution that addresses the problem
4. Provide clear explanations and comments in the code
5. Include error handling and best practices
6. Suggest additional considerations or improvements

Guidelines:
- Write clean, readable, and well-documented code
- Follow best practices for the specified programming language
- Include error handling where appropriate
- Provide step-by-step explanations for complex logic
- Reference the manual sections when using specific APIs or methods
- If the manual sections don't provide enough information, clearly state what additional information would be needed"""

        # User prompt construction
        user_prompt = f"I need help solving the following problem:\n\n"
        user_prompt += f"**Problem Description:**\n{problem_description}\n\n"
        
        if programming_language:
            user_prompt += f"**Target Programming Language:** {programming_language}\n\n"
        
        if technology_stack:
            user_prompt += f"**Technology Stack:** {technology_stack}\n\n"
        
        # Add manual references
        if manual_results and len(manual_results) > 0:
            limited_results = manual_results[:max_manual_refs]
            user_prompt += f"**Relevant Manual Sections ({len(limited_results)} references):**\n\n"
            
            for i, result in enumerate(limited_results, 1):
                chunk = result['chunk']
                relevance = result['relevance_score'] * 100
                
                user_prompt += f"**Reference {i}** (Relevance: {relevance:.1f}%)\n"
                user_prompt += f"Manual: {result['manual_title']}\n"
                user_prompt += f"Section: {result['section_key']}\n"
                if result['chapter']:
                    user_prompt += f"Chapter: {result['chapter']}\n"
                user_prompt += f"Type: {result['section_type']}\n"
                user_prompt += f"Content:\n```\n{chunk['text']}\n```\n\n"
        else:
            user_prompt += "**Note:** No specific manual references were found. Please provide a general solution based on best practices.\n\n"
        
        user_prompt += "Please provide:\n"
        user_prompt += "1. A complete code solution with clear comments\n"
        user_prompt += "2. Step-by-step explanation of the implementation\n"
        user_prompt += "3. Any assumptions made\n"
        user_prompt += "4. Potential edge cases and how to handle them\n"
        user_prompt += "5. Suggestions for testing and validation\n"
        user_prompt += "6. Additional resources or documentation that might be helpful\n"
        
        return {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "model": model_id
        }
    
    def generate_code_solution(self, prompt_info: Dict[str, str], max_tokens: int = 4000) -> str:
        """
        Generate code solution using Claude API
        
        Args:
            prompt_info: Prompt information dictionary
            max_tokens: Maximum tokens to generate
            
        Returns:
            str: Generated code solution
        """
        try:
            system_prompt = prompt_info["system_prompt"]
            user_prompt = prompt_info["user_prompt"]
            model = prompt_info.get("model", "claude-3-5-sonnet-20240620")
            
            logger.info(f"Generating code solution with Claude API ({model})...")
            log_memory_usage("Code generation start")
            
            # Call Claude API
            message = self.client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=0.3,  # Lower temperature for more consistent code generation
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            # Extract response
            result = message.content[0].text
            
            log_memory_usage("Code generation complete")
            logger.info("Code solution generated successfully")
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating code solution: {str(e)}")
            return f"Error: Unable to generate solution. {str(e)}"
    
    def get_available_manuals(self) -> List[str]:
        """Get list of available manuals"""
        if not self.vector_db:
            return []
        
        return list(self.vector_db.chunk_metadata.get('manual_titles', []))
    
    def get_available_section_types(self) -> List[str]:
        """Get list of available section types"""
        if not self.vector_db:
            return []
        
        return list(self.vector_db.chunk_metadata.get('section_types', []))
    
    def analyze_problem_context(self, problem: str) -> Dict[str, Any]:
        """Analyze problem to suggest search parameters"""
        problem_lower = problem.lower()
        
        # Detect programming languages
        languages = ['python', 'java', 'javascript', 'c++', 'c#', 'go', 'rust', 'php', 'ruby', 'sql']
        detected_languages = [lang for lang in languages if lang in problem_lower]
        
        # Detect technologies
        technologies = ['api', 'database', 'web', 'mobile', 'cloud', 'docker', 'kubernetes', 'rest', 'graphql']
        detected_technologies = [tech for tech in technologies if tech in problem_lower]
        
        # Suggest section types
        suggested_sections = []
        if any(keyword in problem_lower for keyword in ['example', 'sample', 'demo']):
            suggested_sections.append('example')
        if any(keyword in problem_lower for keyword in ['api', 'function', 'method']):
            suggested_sections.append('api')
        if any(keyword in problem_lower for keyword in ['config', 'setup', 'install']):
            suggested_sections.append('configuration')
        if any(keyword in problem_lower for keyword in ['error', 'bug', 'issue', 'problem']):
            suggested_sections.append('error')
        
        return {
            'detected_languages': detected_languages,
            'detected_technologies': detected_technologies,
            'suggested_sections': suggested_sections
        }

def check_api_key():
    """Check if API key is set"""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("\nError: Environment variable 'ANTHROPIC_API_KEY' is not set.")
        print("You can set it as follows:")
        if sys.platform.startswith('win'):
            print("  Command Prompt: set ANTHROPIC_API_KEY=your_api_key")
            print("  PowerShell: $env:ANTHROPIC_API_KEY = 'your_api_key'")
        else:
            print("  Bash: export ANTHROPIC_API_KEY=your_api_key")
        
        # Option to input API key directly
        use_input = input("\nWould you like to input the API key directly? (y/n, default: y): ").lower() or "y"
        if use_input == "y":
            api_key = input("Enter your Anthropic API key: ").strip()
            if api_key:
                os.environ["ANTHROPIC_API_KEY"] = api_key
                print("API key set successfully (valid for this session only)")
                return True
        
        return False
    return True

def interactive_mode():
    """Run interactive mode for manual search RAG"""
    print("\n" + "=" * 80)
    print("🔍 Manual Search RAG System - Interactive Mode (Claude API)")
    print("=" * 80)
    
    # Check API key
    if not check_api_key():
        print("API key not set. Exiting program.")
        return
    
    # Create RAG system
    vector_db_dir = input("\nEnter vector database path (default: manual_vector_db): ") or "manual_vector_db"
    
    rag_system = ManualSearchRAG(model_type="claude", vector_db_dir=vector_db_dir)
    
    # Load model
    if not rag_system.load_model():
        print("Failed to set up API client. Exiting program.")
        return
    
    # Load vector database
    if not rag_system.load_vector_db():
        print("Failed to load vector database. Exiting program.")
        return
    
    # Select Claude model
    print("\nSelect Claude model:")
    models = [
        "claude-3-5-sonnet-20240620",
        "claude-3-opus-20240229", 
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307"
    ]
    
    for i, model in enumerate(models, 1):
        print(f"{i}. {model}")
    
    model_choice = input("\nEnter model number (default: 1): ") or "1"
    try:
        model_idx = int(model_choice) - 1
        if 0 <= model_idx < len(models):
            selected_model = models[model_idx]
        else:
            print("Invalid selection. Using first model.")
            selected_model = models[0]
    except ValueError:
        print("Invalid input. Using first model.")
        selected_model = models[0]
    
    print(f"\nSelected model: {selected_model}")
    
    # Display available manuals and section types
    available_manuals = rag_system.get_available_manuals()
    available_sections = rag_system.get_available_section_types()
    
    print(f"\nAvailable manuals ({len(available_manuals)}):")
    for i, manual in enumerate(available_manuals[:10], 1):  # Show first 10
        print(f"  {i}. {manual}")
    if len(available_manuals) > 10:
        print(f"  ... and {len(available_manuals) - 10} more")
    
    print(f"\nAvailable section types: {', '.join(available_sections)}")
    
    while True:
        print("\n" + "=" * 80)
        print("Enter your problem description:")
        print("=" * 80)
        
        # Get problem description
        problem_lines = []
        print("Enter your problem (press Enter twice to finish):")
        while True:
            line = input("> ")
            if line.strip() == "" and problem_lines:
                break
            problem_lines.append(line)
        
        problem_description = "\n".join(problem_lines).strip()
        
        if not problem_description:
            print("No problem description provided.")
            continue
        
        # Analyze problem context
        context = rag_system.analyze_problem_context(problem_description)
        
        print(f"\n📊 Problem Analysis:")
        if context['detected_languages']:
            print(f"Detected languages: {', '.join(context['detected_languages'])}")
        if context['detected_technologies']:
            print(f"Detected technologies: {', '.join(context['detected_technologies'])}")
        if context['suggested_sections']:
            print(f"Suggested section types: {', '.join(context['suggested_sections'])}")
        
        # Get additional parameters
        programming_language = input(f"\nProgramming language (detected: {', '.join(context['detected_languages']) or 'none'}): ").strip()
        if not programming_language and context['detected_languages']:
            programming_language = context['detected_languages'][0]
        
        technology_stack = input("Technology stack (optional): ").strip()
        
        # Search parameters
        try:
            k = int(input("Number of manual sections to search (default: 10): ") or "10")
        except ValueError:
            k = 10
        
        filter_manual = input("Filter by specific manual (optional, exact name): ").strip() or None
        filter_section_type = input("Filter by section type (optional): ").strip() or None
        
        # Search for relevant manual sections
        print(f"\n🔍 Searching for relevant manual sections...")
        start_time = time.time()
        
        manual_results = rag_system.search_manual_solutions(
            problem_description=problem_description,
            programming_language=programming_language,
            technology_stack=technology_stack,
            k=k,
            filter_manual=filter_manual,
            filter_section_type=filter_section_type
        )
        
        search_time = time.time() - start_time
        
        if not manual_results:
            print("No relevant manual sections found.")
            continue_choice = input("Continue without manual references? (y/n): ").lower()
            if continue_choice != 'y':
                continue
        else:
            print(f"\n📚 Found {len(manual_results)} relevant manual sections ({search_time:.2f}s):")
            print("=" * 60)
            
            for i, result in enumerate(manual_results[:5], 1):  # Show top 5
                print(f"\n{i}. [{result['manual_title']}]")
                print(f"   Section: {result['section_key']}")
                print(f"   Type: {result['section_type']}")
                print(f"   Relevance: {result['relevance_score']*100:.1f}%")
                print(f"   Preview: {result['context_summary']}")
            
            if len(manual_results) > 5:
                print(f"\n... and {len(manual_results) - 5} more results")
        
        # Generate solution prompt
        try:
            max_refs = int(input(f"\nMax manual references to include in prompt (default: 5): ") or "5")
        except ValueError:
            max_refs = 5
        
        prompt_info = rag_system.generate_solution_prompt(
            problem_description=problem_description,
            programming_language=programming_language,
            technology_stack=technology_stack,
            manual_results=manual_results,
            max_manual_refs=max_refs,
            model_id=selected_model
        )
        
        # Show prompt preview
        show_prompt = input("\nShow full prompt before sending to Claude? (y/n, default: n): ").lower()
        if show_prompt == 'y':
            print("\n" + "=" * 80)
            print("System Prompt:")
            print("=" * 80)
            print(prompt_info["system_prompt"])
            print("\n" + "=" * 80)
            print("User Prompt:")
            print("=" * 80)
            print(prompt_info["user_prompt"])
        
        # Generate solution
        proceed = input("\nGenerate code solution with Claude API? (y/n, default: y): ").lower() or "y"
        if proceed != 'y':
            continue
        
        try:
            max_tokens = int(input("Max tokens to generate (default: 4000): ") or "4000")
        except ValueError:
            max_tokens = 4000
        
        print(f"\n🤖 Generating code solution with Claude API...")
        print("This may take a moment...")
        
        start_time = time.time()
        solution = rag_system.generate_code_solution(prompt_info, max_tokens=max_tokens)
        generation_time = time.time() - start_time
        
        # Display solution
        print("\n" + "=" * 80)
        print(f"💡 Generated Code Solution (Generation time: {generation_time:.2f}s)")
        print("=" * 80)
        print(solution)
        
        # Save solution
        save_choice = input("\nSave solution to file? (y/n, default: n): ").lower()
        if save_choice == 'y':
            filename = input("Enter filename (default: solution.md): ") or "solution.md"
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"# Code Solution\n\n")
                f.write(f"**Problem:** {problem_description}\n\n")
                if programming_language:
                    f.write(f"**Language:** {programming_language}\n\n")
                if technology_stack:
                    f.write(f"**Technology:** {technology_stack}\n\n")
                f.write(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write("---\n\n")
                f.write(solution)
            
            print(f"Solution saved to: {filename}")
        
        # Continue or exit
        continue_choice = input("\nSolve another problem? (y/n, default: y): ").lower() or "y"
        if continue_choice != 'y':
            break
    
    print("\nThank you for using Manual Search RAG System!")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Manual Search RAG System with Claude API')
    parser.add_argument('--check-key', action='store_true', help='Check Anthropic API key')
    parser.add_argument('--set-key', type=str, help='Set Anthropic API key')
    parser.add_argument('--vector-db', type=str, default='manual_vector_db', help='Vector database directory')
    parser.add_argument('--problem', type=str, help='Problem description for direct solving')
    parser.add_argument('--language', type=str, help='Programming language')
    parser.add_argument('--tech-stack', type=str, help='Technology stack')
    
    args = parser.parse_args()
    
    # Set API key
    if args.set_key:
        os.environ["ANTHROPIC_API_KEY"] = args.set_key
        print("Anthropic API key has been set.")
        return
    
    # Check API key
    if args.check_key:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if api_key:
            masked_key = api_key[:4] + "*" * (len(api_key) - 8) + api_key[-4:]
            print(f"Anthropic API key is set: {masked_key}")
        else:
            print("Anthropic API key is not set.")
        return
    
    # Direct problem solving mode
    if args.problem:
        if not check_api_key():
            print("API key not set. Cannot proceed.")
            return
        
        print("Direct problem solving mode")
        rag_system = ManualSearchRAG(vector_db_dir=args.vector_db)
        
        if not rag_system.load_model() or not rag_system.load_vector_db():
            print("Failed to initialize system.")
            return
        
        # Search and generate solution
        manual_results = rag_system.search_manual_solutions(
            problem_description=args.problem,
            programming_language=args.language or "",
            technology_stack=args.tech_stack or "",
            k=10
        )
        
        prompt_info = rag_system.generate_solution_prompt(
            problem_description=args.problem,
            programming_language=args.language or "",
            technology_stack=args.tech_stack or "",
            manual_results=manual_results
        )
        
        solution = rag_system.generate_code_solution(prompt_info)
        
        print("\nGenerated Solution:")
        print("=" * 80)
        print(solution)
        return
    
    # Interactive mode
    interactive_mode()

if __name__ == "__main__":
    main()