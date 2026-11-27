# Inventário técnico do Vestígio — APIs, bibliotecas, modelos e repositórios

Este documento descreve o que está efetivamente integrado nesta versão do projeto. Ele diferencia **API HTTP**, **biblioteca**, **modelo de IA**, **serviço de distribuição de arquivos/pesos** e **repositório de código** — não são a mesma coisa.

## 1. Resumo: quais APIs externas o Vestígio usa?

**O Vestígio não chama uma API externa de detecção de imagens.** A análise é executada localmente pelo Python, usando os pesos do Smaug baixados do Hugging Face na primeira inicialização. Não há chave de API, token pago ou envio da imagem para um serviço de inferência remoto.

Há, porém, serviços externos usados para obter dependências e recursos:

| Serviço | Para que serve | Envia a imagem analisada? | Precisa ficar online depois do primeiro download? |
|---|---|---:|---:|
| Hugging Face Hub | Baixar arquivos e pesos do modelo Smaug; também hospeda os repositórios citados | Não, no fluxo normal de inferência local | Não, se o modelo já estiver em cache |
| Google Fonts | Carregar as famílias tipográficas Bricolage Grotesque e Source Serif 4 no navegador | Não | Não é essencial; fontes locais substituem se estiver offline |
| API local do próprio Vestígio | Comunicação entre a página e o backend Flask em `127.0.0.1` | A imagem vai do navegador ao próprio computador | Sim, enquanto o app estiver sendo usado |

O navegador envia o arquivo para o backend local em `POST /api/analyze`. O backend decodifica a imagem, extrai sinais de metadados e executa o modelo no computador. A imagem não é encaminhada ao Hugging Face nem ao Google Fonts.

## 2. Modelo de inteligência artificial principal

### Smaug-AI-Image-Detector

- **Identificador:** `abacusai/Smaug-AI-Image-Detector`
- **Publicador:** Abacus.AI
- **Repositório do modelo:** https://huggingface.co/abacusai/Smaug-AI-Image-Detector
- **Cartão do modelo:** https://huggingface.co/abacusai/Smaug-AI-Image-Detector
- **Licença declarada no repositório:** Apache-2.0
- **Arquitetura/base informada pelo publicador:** encoder visual EmbeddingGemma-2 com classificador linear.
- **Tamanho informado:** aproximadamente 439 milhões de parâmetros; o download completo dos arquivos necessários pode ficar próximo de 910 MB, conforme a versão/cache.
- **Papel no Vestígio:** produzir um score numérico para a classe de imagem gerada por IA.
- **Execução:** local, por meio de PyTorch e Sentence-Transformers.

**Como o adaptador do Vestígio usa o modelo:** `backend/smaug_model.py` corrige a orientação EXIF, converte a imagem para RGB, prepara três recortes quadrados de 528 × 528 pixels e envia esses recortes ao modelo. O código calcula a média dos logits dos recortes e aplica sigmoid para obter um score entre 0 e 1.

O score não é uma probabilidade calibrada de que a imagem seja artificial nem uma medida garantida de acerto. Os limites de 20% e 80% usados para redigir o veredito são convenções do aplicativo, não limites de precisão comprovada.

### Código de referência do Smaug

- **Arquivo de referência:** `smaug_image_detect.py`
- **Link direto:** https://huggingface.co/abacusai/Smaug-AI-Image-Detector/blob/main/smaug_image_detect.py
- **Uso no Vestígio:** referência para o pré-processamento e a inferência. A implementação foi adaptada e mantida no próprio arquivo `backend/smaug_model.py`; o aplicativo não baixa e executa automaticamente um script Python remoto.

### Modelo legado opcional

- **Identificador:** `Ateeqq/ai-vs-human-image-detector`
- **Repositório:** https://huggingface.co/Ateeqq/ai-vs-human-image-detector
- **Uso:** compatibilidade para comparação manual, se for incluído em `MODEL_IDS` no `.env`.
- **Estado:** não é o modelo padrão desta versão. É um classificador legado carregado pelo caminho genérico de Transformers.

## 3. Bibliotecas e frameworks Python

As versões mínimas declaradas estão em `requirements.txt`. O instalador pode resolver versões mais recentes compatíveis; para reprodutibilidade rigorosa, seria necessário criar também um arquivo de lock com versões exatas.

| Dependência | Papel no projeto | Onde é usada | Repositório / documentação |
|---|---|---|---|
| Flask (`flask>=3.0`) | Servidor web local, rotas HTTP, respostas JSON e entrega da página | `backend/app.py` | https://github.com/pallets/flask · https://flask.palletsprojects.com/ |
| Pillow (`pillow>=10.3`) | Abrir, validar, converter e redimensionar imagens; ler EXIF e metadados PNG | `backend/imaging.py`, `backend/metadata.py`, `backend/smaug_model.py` | https://github.com/python-pillow/Pillow · https://pillow.readthedocs.io/ |
| python-dotenv (`python-dotenv>=1.0`) | Ler configurações opcionais do arquivo `.env` | `backend/config.py` | https://github.com/theskumar/python-dotenv |
| PyTorch (`torch>=2.7`) | Tensor, execução do modelo, inferência sem gradientes, sigmoid e suporte a CPU/CUDA | `backend/detector.py`, `backend/smaug_model.py` | https://github.com/pytorch/pytorch · https://pytorch.org/docs/ |
| Transformers (`transformers>=5.20.0`) | Compatibilidade com modelos de classificação de imagem configurados como alternativa | `backend/detector.py` | https://github.com/huggingface/transformers · https://huggingface.co/docs/transformers/ |
| Sentence-Transformers (`sentence-transformers[image]>=5.2.0`) | Carregar o Smaug e executar a codificação/inferência de imagens | `backend/smaug_model.py` | https://github.com/UKPLab/sentence-transformers · https://sbert.net/ |

**Observação sobre Transformers:** o caminho principal do Smaug usa Sentence-Transformers. Transformers permanece como compatibilidade para modelos legados configurados manualmente. O `requirements.txt` mantém as duas dependências porque ambas são usadas pelo código.

## 4. Frontend e recursos do navegador

O frontend não usa React, Vue, Angular, jQuery ou outro framework JavaScript. É feito com tecnologias nativas:

- **HTML5:** estrutura da interface (`frontend/index.html`).
- **CSS3:** layout, responsividade e identidade visual (`frontend/css/styles.css`).
- **JavaScript puro (Vanilla JS):** seleção e envio do arquivo, chamadas `fetch` à API local e atualização da interface (`frontend/js/app.js`).
- **Google Fonts:** Bricolage Grotesque e Source Serif 4, solicitadas pelo navegador em `fonts.googleapis.com` e servidas por `fonts.gstatic.com`. São um recurso visual externo, não uma API de detecção.

Repositórios/documentação das tecnologias nativas:
- HTML: https://developer.mozilla.org/docs/Web/HTML
- CSS: https://developer.mozilla.org/docs/Web/CSS
- JavaScript: https://developer.mozilla.org/docs/Web/JavaScript
- Fetch API: https://developer.mozilla.org/docs/Web/API/Fetch_API

## 5. API HTTP interna do Vestígio

Esta é uma API REST local criada pelo próprio projeto. Ela não é um serviço de terceiro.

### `GET /api/health`

Retorna o estado do motor de detecção e os limites aceitos pelo servidor. É útil para a interface mostrar se o modelo está carregando, pronto ou indisponível.

Exemplo ilustrativo de resposta (os valores variam conforme a configuração):

```json
{
  "status": "ok",
  "engine": {
    "state": "ready",
    "device": "cpu",
    "models": []
  },
  "limits": {
    "max_upload_bytes": 10485760,
    "mime_types": ["image/jpeg", "image/png", "image/webp"],
    "extensions": [".jpg", ".jpeg", ".png", ".webp"],
    "min_side": 64
  }
}
```

### `POST /api/analyze`

- **Formato:** `multipart/form-data`
- **Campo obrigatório:** `image`
- **Formatos aceitos:** JPEG, PNG e WEBP.
- **Limite padrão de arquivo:** 10 MiB.
- **Dimensão mínima:** 64 pixels no menor lado.
- **Limite de pixels decodificados:** 40 milhões por padrão.
- **Resposta:** JSON com `verdict`, `models`, `metadata.signals`, dimensões/formato/tamanho da imagem, notas explicativas, aviso de limitação e tempo da análise.

Erros relevantes usam códigos HTTP como 400 (arquivo ausente), 413 (arquivo grande), 429 (motor ocupado), 503 (modelo carregando/indisponível), 502 (resposta inesperada do detector) e 500 (falha interna).

## 6. Sinais de metadados: o que são e o que não são

O módulo `backend/metadata.py` usa Pillow e leitura de bytes para procurar sinais complementares:

- Marcador IPTC `trainedAlgorithmicMedia` e marcador relacionado.
- Texto/metadata de PNG com parâmetros ou workflow de ferramentas como Stable Diffusion e ComfyUI.
- Menções textuais a ferramentas conhecidas de geração (por exemplo, Midjourney, DALL·E, NovelAI, Fooocus e Adobe Firefly).
- EXIF de câmera, fabricante/modelo e alguns campos de captura.
- Presença aparente de dados C2PA no arquivo.

**Limitação importante:** o Vestígio não valida criptograficamente uma credencial C2PA. A presença de bytes relacionados a C2PA é apenas sinal informativo; para validar emissor e integridade é necessário um verificador compatível. Metadados podem ser removidos, alterados ou copiados. Não são prova isolada de autoria ou origem.

Referências dos padrões:
- IPTC Photo Metadata: https://www.iptc.org/standards/photo-metadata/
- C2PA: https://c2pa.org/ e https://spec.c2pa.org/
- EXIF (biblioteca de leitura usada pelo projeto): https://pillow.readthedocs.io/en/stable/reference/ExifTags.html

## 7. Repositórios e páginas oficiais — lista consolidada

| Projeto / recurso | Link | Relação com o Vestígio |
|---|---|---|
| Smaug-AI-Image-Detector | https://huggingface.co/abacusai/Smaug-AI-Image-Detector | Modelo principal e pesos |
| Código de inferência de referência do Smaug | https://huggingface.co/abacusai/Smaug-AI-Image-Detector/blob/main/smaug_image_detect.py | Referência técnica adaptada em `backend/smaug_model.py` |
| Licença do Smaug | https://huggingface.co/abacusai/Smaug-AI-Image-Detector/blob/main/LICENSE | Termos publicados para o modelo |
| Ateeqq AI-vs-Human (legado) | https://huggingface.co/Ateeqq/ai-vs-human-image-detector | Alternativa opcional, não padrão |
| Flask | https://github.com/pallets/flask | Servidor e API local |
| Pillow | https://github.com/python-pillow/Pillow | Processamento e leitura de imagens |
| python-dotenv | https://github.com/theskumar/python-dotenv | Configuração `.env` |
| PyTorch | https://github.com/pytorch/pytorch | Motor de inferência tensorial |
| Transformers | https://github.com/huggingface/transformers | Compatibilidade com classificadores legados |
| Sentence-Transformers | https://github.com/UKPLab/sentence-transformers | Carregamento e inferência do Smaug |
| Python | https://github.com/python/cpython | Linguagem/runtime do backend |
| Documentação HTML | https://developer.mozilla.org/docs/Web/HTML | Frontend |
| Documentação CSS | https://developer.mozilla.org/docs/Web/CSS | Frontend |
| Documentação JavaScript | https://developer.mozilla.org/docs/Web/JavaScript | Frontend |
| Google Fonts | https://fonts.google.com/ | Fontes tipográficas externas |
| IPTC Photo Metadata | https://www.iptc.org/standards/photo-metadata/ | Referência para metadados de origem digital |
| C2PA | https://c2pa.org/ | Referência para credenciais de conteúdo |

## 8. Arquitetura: fluxo de ponta a ponta

```text
Pessoa seleciona imagem no navegador
             |
             v
Frontend HTML/CSS/JavaScript
             |
             | POST /api/analyze (multipart/form-data)
             v
Flask — backend/app.py
             |
             +--> valida tamanho e decodifica com Pillow
             |
             +--> extrai sinais de EXIF / PNG / IPTC / C2PA
             |
             +--> DetectionEngine (backend/detector.py)
                        |
                        v
                 Smaug adapter
                 backend/smaug_model.py
                        |
                        v
                 Sentence-Transformers + PyTorch
                        |
                        v
                 Pesos locais do modelo Smaug
                        |
                        v
                 Score IA (0 a 1)
             |
             v
backend/scoring.py combina score e metadados
             |
             v
Resposta JSON local -> frontend exibe o resultado
```

## 9. Configurações importantes

As opções são lidas de variáveis de ambiente e, opcionalmente, do `.env` (modelo de arquivo: `.env.example`).

| Variável | Padrão | Função |
|---|---|---|
| `MODEL_IDS` | `abacusai/Smaug-AI-Image-Detector` | Um ou mais IDs do Hugging Face separados por vírgula; até 3 modelos |
| `DETECTOR_DEVICE` | `auto` | `auto`, `cpu` ou `cuda` |
| `MAX_UPLOAD_MB` | `10` | Tamanho máximo de upload em MiB |
| `MAX_IMAGE_PIXELS` | `40000000` | Limite de pixels da imagem decodificada |
| `APP_HOST` | `127.0.0.1` | Endereço de escuta; padrão restrito ao próprio computador |
| `APP_PORT` | `5000` | Porta HTTP local |

O primeiro carregamento precisa de internet para baixar os arquivos do modelo. Depois do cache, o modo offline pode ser usado com a variável `HF_HUB_OFFLINE=1`.

## 10. Segurança, privacidade e limites

- Não há API key nem credencial externa no projeto.
- No fluxo normal, a imagem é enviada apenas ao backend local e processada em memória.
- O servidor verifica o conteúdo real da imagem, não confia somente no nome/extensão ou no `Content-Type` informado pelo navegador.
- Há limites de bytes e pixels para reduzir risco de arquivos excessivamente grandes.
- O backend adiciona CSP, `X-Content-Type-Options`, `X-Frame-Options`, `Referrer-Policy` e `Cache-Control: no-store` nas rotas de API.
- Erros internos são registrados no terminal com referência de requisição; detalhes técnicos não são enviados ao navegador.
- O padrão `127.0.0.1` limita o servidor ao próprio computador. Não exponha o servidor de desenvolvimento diretamente à internet.
- O modelo pode errar, inclusive com score alto. Não use o resultado sozinho para acusar alguém, autenticar evidências ou tomar decisões importantes.
- A avaliação de desempenho mencionada no README é a publicada pelo mantenedor do modelo; não é benchmark independente do projeto.

## 11. O que não está integrado

Para evitar interpretações erradas, esta versão **não** integra:

- API paga ou remota de detecção (Hive, Sightengine, Illuminarty ou similares).
- API da OpenAI, Gemini, Claude ou outro chatbot para classificar imagens.
- Banco de dados, autenticação de usuários, contas ou armazenamento de histórico.
- Verificador criptográfico C2PA completo.
- Serviço de treinamento/fine-tuning próprio.

Esses serviços não devem ser listados como APIs usadas pelo Vestígio atual. São possibilidades futuras, não componentes desta versão.

## 12. Como conferir no código

- Dependências instaladas: `requirements.txt`.
- IDs de modelos e limites: `backend/config.py` e `.env.example`.
- Carregamento e inferência: `backend/detector.py` e `backend/smaug_model.py`.
- Rotas HTTP: `backend/app.py`.
- Processamento de imagem: `backend/imaging.py`.
- Metadados: `backend/metadata.py`.
- Interpretação do score: `backend/scoring.py`.
- Frontend e chamadas HTTP: `frontend/index.html`, `frontend/js/app.js`, `frontend/css/styles.css`.
- Instalação e execução: `README.md`.
