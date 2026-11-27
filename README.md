# Vestígio

Site local que estima se uma imagem se parece com as classes **gerada por IA** ou **humana**.
Backend em Python (Flask), frontend em HTML, CSS e JavaScript puros, detecção feita por um modelo
de visão computacional que roda **no seu computador**: as imagens não são enviadas a terceiros.

## Como a detecção funciona

- **Modelo padrão:** [`abacusai/Smaug-AI-Image-Detector`](https://huggingface.co/abacusai/Smaug-AI-Image-Detector), da Abacus.AI, baseado no encoder visual EmbeddingGemma-2. Licença Apache-2.0.
- **Execução local:** depois de baixar os pesos na primeira execução, a inferência ocorre no computador. As imagens analisadas não são enviadas para uma API externa. O download inicial do modelo tem aproximadamente 910 MB; o carregamento em CPU pode exigir memória adicional significativa.
- **Implementação:** o adaptador de inferência usado pelo Vestígio fica versionado em `backend/smaug_model.py`. O aplicativo não importa nem executa automaticamente um arquivo Python remoto baixado do repositório do modelo.
- **Avaliação publicada:** o cartão do modelo informa AUC média de 0,9816 em três conjuntos de avaliação, com 26.070 imagens no total, e taxas de fotos reais sinalizadas como IA de 3,0%, 1,4% e 0,4% nos três conjuntos. Esses números são publicados pelo mantenedor do modelo; não são uma avaliação independente feita pelo Vestígio e não garantem o mesmo desempenho nas imagens do usuário.
- **Score:** o Smaug retorna um score entre 0 e 1 para a classe IA, obtido pela média dos logits de três recortes e aplicação de sigmoid. Esse score **não é uma probabilidade calibrada de acerto**.
- **Veredito do Vestígio:** abaixo de 20% = «mais semelhante à classe humana»; acima de 80% = «mais semelhante à classe IA»; entre esses valores = «inconclusivo». São faixas conservadoras de apresentação, não limites validados com um conjunto próprio.
- **Modelo antigo:** `Ateeqq/ai-vs-human-image-detector` permanece disponível para comparação manual via `MODEL_IDS`, mas não é mais o padrão. O cartão do Smaug compara os modelos em três conjuntos e reporta resultados inferiores e maior taxa de falsos positivos para o Ateeqq.
- **Metadados (separados do score):** o backend procura declarações explícitas de IA (marcador IPTC `trainedAlgorithmicMedia`, parâmetros de Stable Diffusion/ComfyUI gravados em PNG) e dados EXIF de câmera. Metadados podem ser apagados ou forjados; sua ausência não prova nada.
- **Limitações:** geradores novos, imagens manipuladas e conteúdos híbridos podem enganar o detector. Nenhum detector prova a origem de uma imagem; o resultado não deve ser usado sozinho para acusar alguém ou tomar decisões importantes.

Fonte dos dados de avaliação do Smaug: [cartão oficial do modelo no Hugging Face](https://huggingface.co/abacusai/Smaug-AI-Image-Detector).

## Estrutura

```
vestigio/
├── run.py                 # inicia o servidor
├── requirements.txt
├── .env.example           # configuração opcional
├── .gitignore
├── backend/
│   ├── app.py             # Flask: rotas, validação, cabeçalhos de segurança
│   ├── config.py          # leitura de variáveis de ambiente
│   ├── detector.py        # gerencia os modelos e roda a inferência
│   ├── smaug_model.py     # recortes e adaptador local do Smaug
│   ├── imaging.py         # valida e decodifica a imagem (em memória)
│   ├── metadata.py        # sinais de EXIF / PNG / IPTC / C2PA
│   └── scoring.py         # normaliza a resposta e escreve o veredito
├── frontend/
│   ├── index.html
│   ├── favicon.svg
│   ├── css/styles.css
│   └── js/app.js
└── tests/test_api.py      # testes automatizados (motor falso)
```

## Instalação no Windows (PowerShell no VS Code)

**Você precisa de:** Windows 10/11 de 64 bits, vários GB de RAM disponíveis (o Smaug é um modelo de 439 milhões de parâmetros), espaço para baixar aproximadamente 910 MB de pesos e internet na primeira execução,
[Python 3.12 ou 3.13 de 64 bits](https://www.python.org/downloads/windows/) (na instalação, marque **Add python.exe to PATH**)
e o [Visual Studio Code](https://code.visualstudio.com/).

1. Verifique o Python. No PowerShell:
   ```powershell
   python --version
   pip --version
   ```
   Se `python` não for reconhecido, tente `py --version`; se funcionar, use `py` no lugar de `python` nos comandos abaixo.
2. Extraia a pasta `vestigio` (ou crie os arquivos conforme a árvore acima) e abra-a: no VS Code, **Arquivo > Abrir Pasta...**.
3. Abra o terminal: **Terminal > Novo Terminal** (será PowerShell; confira que o caminho termina em `\vestigio`).
4. Crie e ative o ambiente virtual:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
   O prompt deve passar a começar com `(.venv)`. Se der erro de "execução de scripts desabilitada":
   ```powershell
   Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
   ```
   e repita a ativação. No `cmd` o comando é `.venv\Scripts\activate.bat`.
   Se o VS Code perguntar qual interpretador usar, escolha o de `.venv`.
5. Instale as dependências (baixa PyTorch, Transformers e Sentence-Transformers; pode levar vários minutos):
   ```powershell
   python -m pip install --upgrade pip
   python -m pip install --upgrade -r requirements.txt
   ```
6. (Opcional) Configure: `Copy-Item .env.example .env` e edite o `.env`. Não há chave de API.

## Execução

```powershell
python run.py
```

Abra **http://127.0.0.1:5000** no navegador. No canto superior esquerdo há um indicador do motor:

- amarelo piscando, «Carregando o modelo…»: na primeira vez o modelo é baixado (algumas centenas de MB). Veja o progresso no terminal.
- verde, «Motor pronto»: pode analisar.
- vermelho: o modelo não carregou; o motivo aparece no terminal.

Para encerrar, pressione `Ctrl+C` no terminal. Os pesos ficam no cache do Hugging Face em `C:\Users\<você>\.cache\huggingface`. O primeiro carregamento pode demorar; em CPU, a inferência também pode ser lenta.

## Testes

**Automáticos** (não precisam de internet nem do modelo; usam um motor falso):
```powershell
python -m unittest discover -s tests -t . -v
```
Cobrem: imagem válida, arquivo que não é imagem, arquivo grande, imagem pequena/corrompida, motor carregando/indisponível/ocupado,
resposta inesperada do detector, erro interno sem vazamento de detalhes, metadados e cabeçalhos de segurança.

**Roteiro manual** (exige o download e a inferência do modelo real):

| # | Faça | Esperado |
|---|------|----------|
| 1 | Abra `http://127.0.0.1:5000/api/health` | JSON com `"state": "ready"` |
| 2 | Envie uma foto real cuja origem você conhece | O score pode variar; registre o resultado para sua avaliação local |
| 3 | Envie uma imagem gerada por IA cuja origem você conhece | Idem; registre o resultado e não trate um único teste como prova de precisão |
| 4 | Renomeie um `.txt` para `.png` e envie | Recusado: «Formato não suportado» / «não é uma imagem reconhecida» |
| 5 | Envie um arquivo maior que 10 MB | Recusado pelo limite de tamanho |
| 6 | Clique em «Analisar outra imagem» | Volta ao início sem recarregar a página |
| 7 | Com o servidor rodando, pare-o (`Ctrl+C`) e clique em analisar | «Sem conexão com o servidor» |
| 8 | Rode com `$env:MODEL_IDS="dono/nao-existe"; python run.py` | Indicador vermelho «Motor indisponível»; erro no terminal |
| 9 | Reduza a janela do navegador (ou use F12 > modo dispositivo) | Layout vira uma coluna, sem rolagem horizontal |
| 10 | Depois do download, ponha `HF_HUB_OFFLINE=1` no `.env`, desligue a internet e reinicie | Continua funcionando |

> Não há chave de API neste projeto, então «chave ausente ou inválida» corresponde ao teste 8 (modelo que não carrega).

## Problemas comuns

| Sintoma | Causa e solução |
|---------|-----------------|
| `python` não é reconhecido | Reinstale marcando «Add python.exe to PATH», ou use `py`. |
| `Activate.ps1` não pode ser carregado | Rode o `Set-ExecutionPolicy` do passo 4. |
| `No matching distribution found for torch` | Python de 32 bits ou versão nova demais. Use Python 3.12 ou 3.13 de 64 bits. |
| `DLL load failed` / `WinError 126` ao importar torch | Instale o *Microsoft Visual C++ Redistributable 2015-2022 (x64)* e reinicie o terminal. |
| `ModuleNotFoundError: backend` | Rode `python run.py` na pasta raiz do projeto, com o `.venv` ativo. |
| `Address already in use` | Outra coisa usa a porta 5000. Defina `APP_PORT=5001` no `.env`. |
| Indicador vermelho logo no início | Sem internet no primeiro download, firewall/proxy bloqueando `huggingface.co`, ou dependências não instaladas. Leia o terminal. |
| Fontes diferentes do esperado | As fontes vêm do Google Fonts; sem internet o navegador usa fontes do sistema. Nada quebra. |
| Primeira análise lenta | Normal na CPU; as seguintes costumam ser mais rápidas. |

## Segurança e privacidade

- Credenciais: nenhuma é necessária. Configurações ficam em variáveis de ambiente/`.env` (ignorado pelo Git).
- A imagem é lida em memória, analisada e descartada; **nada é gravado em disco** pelo aplicativo (o Werkzeug pode usar um arquivo temporário do sistema para uploads grandes, apagado ao fim da requisição).
- Validações duplicadas: o navegador confere tipo e tamanho; o servidor confere o **conteúdo** com o Pillow (não confia em extensão nem em `Content-Type`), limita bytes, pixels e tamanho mínimo.
- Erros internos nunca devolvem caminhos ou *tracebacks*: o detalhe fica só no log do terminal.
- Cabeçalhos: CSP restritiva, `nosniff`, `X-Frame-Options`, `no-referrer`. O servidor escuta só em `127.0.0.1`.
- O servidor embutido do Flask é para uso local. Para expor na internet, use um servidor WSGI de produção e adicione autenticação e limite de requisições.

## API

- `GET /api/health`: estado do motor e limites.
- `POST /api/analyze`: `multipart/form-data` com o campo `image`. Devolve `verdict`, `models[]` (com `ai_probability` e `raw_labels`), `metadata.signals[]`, `image`, `notes`, `disclaimer`. Por compatibilidade, `ai_probability` continua sendo o nome do campo; com o Smaug, ele contém o score sigmoid retornado pelo modelo, não uma probabilidade calibrada de acerto.
  Erros: `{"error": {"code", "message"}}` com códigos como `FILE_TOO_LARGE`, `UNSUPPORTED_TYPE`, `INVALID_IMAGE`, `ENGINE_LOADING`, `ENGINE_UNAVAILABLE`, `BUSY`, `DETECTOR_BAD_RESPONSE`.
