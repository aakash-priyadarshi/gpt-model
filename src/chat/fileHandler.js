const multer = require('multer');
const fs = require('fs');
const pdfParse = require('pdf-parse');
const textract = require('textract');
const { createWorker } = require('tesseract.js');
const path = require('path');


// Configure storage for multer
const storage = multer.diskStorage({
    destination: function (req, file, cb) {
        cb(null, path.join(__dirname, '../../uploads')); // Correct path to the uploads folder
    },
    filename: function (req, file, cb) {
        cb(null, file.fieldname + '-' + Date.now() + path.extname(file.originalname));
    }
});


const upload = multer({ storage: storage });

// Asynchronous function to extract text from various file types
async function extractText(filePath, fileType) {
    try {
        if (fileType.includes('pdf')) {
            const buffer = fs.readFileSync(filePath);
            const data = await pdfParse(buffer);
            return data.text;
        } else if (fileType.includes('word') || fileType.includes('officedocument')) {
            return await new Promise((resolve, reject) => {
                textract.fromFileWithPath(filePath, (err, text) => err ? reject(err) : resolve(text));
            });
        } else if (fileType.includes('image')) {
            const worker = await createWorker(); // Workers now come pre-loaded, with language pre-loaded, and pre-initialized
            const { data: { text } } = await worker.recognize(filePath);
            await worker.terminate();
            return text;
        }
    } catch (error) {
        console.error('Failed to extract text from file:', error);
        return null;
    } finally {
        fs.unlinkSync(filePath); // Always clean up the file
    }
}


module.exports = { upload, extractText };
