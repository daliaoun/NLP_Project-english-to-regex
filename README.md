# NL2Regex website

A static site (HTML, CSS, JavaScript, no build step, no server). The two trained models run in the browser.
Tabs: Try it, Report, Slides, Downloads.

## Deploy on Vercel

Option A, command line (from this folder):

    npm i -g vercel
    vercel            # answer the prompts; framework "Other", no build command, output directory "."
    vercel --prod

Option B, GitHub: push this folder (or the whole project and set "Root Directory" to `site`), then
"Add New Project" on vercel.com and import it. Framework preset: Other. Leave build command empty.

Option C, drag and drop: unzip and use the Vercel dashboard "Deploy" with this folder.

## Run locally

    python3 -m http.server 8000      # then open http://localhost:8000
(opening index.html directly with file:// does not work, browsers block loading the model files)

## Contents

    index.html, css/, js/app.js      interface
    js/model.js                      LSTM encoder-decoder + attention inference (PyTorch weights)
    js/regex.js                      matcher for the NL-RX regex dialect (& and ~)
    data/                            exported weights (.bin), vocabularies, 999 test examples
    report/                          technical report (web page and PDF)
    slides/                          presentation viewer and PDF
    downloads/                       source code zip
