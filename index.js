const express = require('express');
const multer = require('multer');
const fs = require('fs');
const pdf = require('pdf-parse');
const mammoth = require('mammoth');
const { spawn } = require('child_process');
const path = require('path');

// Initialize Express app
const app = express();
const port = 3000;

// Set up multer for file uploads
const upload = multer({ dest: 'uploads/' });

app.use(express.static('public'));
app.set('view engine', 'ejs');

// Serve the upload page
app.get('/', (req, res) => {
  res.render('index');
});

// Extract text from PDF file
function extractTextFromPDF(filePath) {
  const data = fs.readFileSync(filePath);
  return pdf(data).then(pdfData => pdfData.text);
}

// Extract text from Word document
function extractTextFromWord(filePath) {
  return mammoth.extractRawText({ path: filePath })
    .then(result => result.value);
}

// Handle file uploads and semantic clustering
app.post('/upload', upload.array('files'), async (req, res) => {
  const files = await Promise.all(req.files.map(async (f) => {
    let text = '';
    if (f.mimetype === 'application/pdf') {
      text = await extractTextFromPDF(f.path);
    } else if (f.mimetype === 'application/vnd.openxmlformats-officedocument.wordprocessingml.document') {
      text = await extractTextFromWord(f.path);
    } else if (f.mimetype === 'text/plain') {
      text = fs.readFileSync(f.path, 'utf-8');
    }

    const txtPath = f.path + '.txt';
    fs.writeFileSync(txtPath, text, 'utf8');  // Store text in a .txt file for processing

    return { path: txtPath, name: f.originalname };
  }));

  const input = JSON.stringify(files);

  const python = spawn('python', ['cluster.py']);
  let output = '', errorOutput = '';

  python.stdout.on('data', (data) => { output += data.toString(); });
  python.stderr.on('data', (data) => { errorOutput += data.toString(); });

  python.on('close', (code) => {
    if (code !== 0) {
      console.error(errorOutput);
      return res.status(500).send('Clustering error');
    }
    const result = JSON.parse(output);
    res.render('result', {
      eps: result.eps.toFixed(4),
      min_samples: result.min_samples,
      keywords: result.keywords,
      clusters: result.clusters
    });
  });

  python.stdin.write(input);
  python.stdin.end();
});

app.listen(port, () => {
  console.log(`Server running on port ${port}`);
});
