const express = require('express');
const multer = require('multer');
const fs = require('fs');
const { spawn } = require('child_process');
const path = require('path');

// Initialize Express app
const app = express();
const port = 3000;

// Set up multer for file uploads
const upload = multer({ dest: 'uploads/' });

app.use(express.static('public'));  // For serving static files
app.set('view engine', 'ejs');  // Set EJS as the templating engine

// Serve the upload page
app.get('/', (req, res) => {
  res.render('index');
});

// Handle file uploads and semantic clustering
app.post('/upload', upload.array('files'), (req, res) => {
  console.log('Files uploaded:', req.files);

  // Build an array of { path, name }
  const files = req.files.map(f => ({
    path: f.path,
    name: f.originalname
  }));

  const input = JSON.stringify(files);

  const python = spawn('python', ['cluster.py']);
  let output = '', errorOutput = '';

  python.stdout.on('data', d => output += d);
  python.stderr.on('data', d => errorOutput += d);

  python.on('close', code => {
    if (code !== 0) {
      console.error(errorOutput);
      return res.status(500).send('Clustering error');
    }
    const result = JSON.parse(output);
    res.render('result', {
      eps: result.eps.toFixed(4),
      min_samples: result.min_samples,
      keywords: result.keywords, // Pass the extracted keywords
      clusters: result.clusters
    });
  });

  python.stdin.write(input);
  python.stdin.end();
});

// Start the server
app.listen(port, () => {
  console.log(`Server running on port ${port}`);
});
