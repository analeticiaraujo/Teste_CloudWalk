import os
import pickle
from dotenv import load_dotenv
import requests
from urllib.parse import urljoin, urlparse

load_dotenv()

from langchain_community.document_loaders import WebBaseLoader
import bs4

# Define o USER_AGENT a partir da variável de ambiente ou usa um valor padrão
USER_AGENT = os.getenv("USER_AGENT", "CloudWalkBot/1.0")

# Função para extrair links de uma página HTML
# Adicionada para evitar links duplicados e garantir que são URLs absolutas
def extract_links(html_content, base_url):
    """
    Extrai todos os links internos de uma página HTML, garantindo que são URLs absolutas.
    """
    soup = bs4.BeautifulSoup(html_content, 'html.parser')
    links = set() # Usamos um set para evitar links duplicados
    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href']
        full_url = urljoin(base_url, href) # Converte URLs relativas em absolutas

        # Opcional: Filtrar links para garantir que permanecemos no domínio da CloudWalk/InfinitePay
        # Isso é CRUCIAL para não sair rastreando a internet inteira!
        parsed_base = urlparse(base_url)
        parsed_full = urlparse(full_url)

        # Checa se o netloc (domínio) é o mesmo ou um subdomínio relacionado
        if parsed_full.netloc == parsed_base.netloc or \
           parsed_full.netloc.endswith(f".{parsed_base.netloc}") or \
           parsed_base.netloc.endswith(f".{parsed_full.netloc}"): # Adicionado para cobrir infinitepay.io e cloudwalk.io

            # Filtra links de email, telefone, etc.
            if full_url.startswith("http") and "#" not in full_url: # Ignora âncoras internas
                links.add(full_url)
    return list(links) # Retorna como lista para facilitar a iteração, se necessário

def load_cloudwalk_data():
    """
    Loads content from CloudWalk websites using LangChain's WebBaseLoader
    and performs a basic link crawling.
    """
    initial_urls = [
        "https://www.cloudwalk.io",
        "https://www.infinitepay.io/tap"
        ]

    visited_urls = set()
    urls_to_visit = list(initial_urls) # Começa com as URLs iniciais
    all_documents = []

    print("Starting web crawling process...")
    while urls_to_visit:
        current_url = urls_to_visit.pop(0) # Pega a próxima URL da fila

        if current_url in visited_urls:
            continue # Já visitamos essa URL, pula!

        print(f"Visiting: {current_url}")
        visited_urls.add(current_url)

        try:
            # Usamos requests para obter o conteúdo HTML bruto para extrair links
            headers = {"User-Agent": USER_AGENT}
            response = requests.get(current_url, headers=headers, timeout=10)
            response.raise_for_status() # Lança um erro para status de erro HTTP

            html_content = response.text
            # Extrai links da página atual
            new_links = extract_links(html_content, current_url)

            for link in new_links:
                if link not in visited_urls and link not in urls_to_visit:
                    urls_to_visit.append(link)

            # Agora usamos o WebBaseLoader para carregar o conteúdo da página de forma estruturada (como Document)
            # para o nosso chatbot, usando as opções de parsing que já tínhamos.
            # Atenção: Isso fará uma segunda requisição para a mesma URL se o WebBaseLoader não tiver um cache interno.
            # Para otimização avançada, poderíamos passar o html_content diretamente para um BeautifulSoup loader na LangChain,
            # mas o WebBaseLoader simplifica a criação do Documento LangChain.
            loader = WebBaseLoader(
                web_paths=[current_url], # Carrega apenas a URL atual
                bs_kwargs={
                    "parse_only": bs4.SoupStrainer(
                        ["p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "span", "div"]
                    )
                },
                requests_kwargs={"headers": {"User-Agent": USER_AGENT}}
            )
            docs = loader.load()
            all_documents.extend(docs) # Adiciona os documentos carregados à lista total

        except requests.exceptions.RequestException as e:
            print(f"Error visiting {current_url}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred with {current_url}: {e}")

    # Filtrar documentos vazios ou com pouco conteúdo no final
    initial_doc_count = len(all_documents)
    all_documents = [doc for doc in all_documents if len(doc.page_content.strip()) > 100]
    print(f"Loaded {initial_doc_count} documents. Filtered to {len(all_documents)} non-empty documents.")

    if all_documents:
        print("\n--- Sample of first loaded document content ---")
        print(all_documents[0].page_content[:500])
        print("---------------------------------------------\n")

    return all_documents

if __name__ == "__main__":
    cloudwalk_docs = load_cloudwalk_data()
    with open("cloudwalk_documents.pkl", "wb") as f:
        pickle.dump(cloudwalk_docs, f)
    print("Data loading complete. Proceed to chunking and embedding.")