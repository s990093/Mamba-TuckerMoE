const fileInput1 = document.getElementById('file-1');
const fileInput2 = document.getElementById('file-2');
const dropZone1 = document.getElementById('drop-zone-1');
const dropZone2 = document.getElementById('drop-zone-2');
const img1 = document.getElementById('img-1');
const img2 = document.getElementById('img-2');
const mergeBtn = document.getElementById('merge-btn');
const resetBtn = document.getElementById('reset-btn');
const downloadBtn = document.getElementById('download-btn');
const canvas = document.getElementById('result-canvas');
const previewContainer = document.getElementById('preview-container');

let images = { img1: null, img2: null };

// Handle file selection
function handleFile(file, index) {
    if (file && file.type.startsWith('image/')) {
        const reader = new FileReader();
        reader.onload = (e) => {
            const imgElement = document.getElementById(`img-${index}`);
            const dropZone = document.getElementById(`drop-zone-${index}`);
            imgElement.src = e.target.result;
            dropZone.classList.add('has-image');
            
            const img = new Image();
            img.src = e.target.result;
            img.onload = () => {
                images[`img${index}`] = img;
            };
        };
        reader.readAsDataURL(file);
    }
}

// Drop zone events
[dropZone1, dropZone2].forEach((zone, i) => {
    const index = i + 1;
    const input = document.getElementById(`file-${index}`);

    zone.addEventListener('click', () => input.click());

    zone.addEventListener('dragover', (e) => {
        e.preventDefault();
        zone.classList.add('active');
    });

    zone.addEventListener('dragleave', () => {
        zone.classList.remove('active');
    });

    zone.addEventListener('drop', (e) => {
        e.preventDefault();
        zone.classList.remove('active');
        const file = e.dataTransfer.files[0];
        handleFile(file, index);
    });

    input.addEventListener('change', (e) => {
        const file = e.target.files[0];
        handleFile(file, index);
    });
});

// Merge Logic
mergeBtn.addEventListener('click', () => {
    if (!images.img1 || !images.img2) {
        alert('請先上傳兩張照片！');
        return;
    }

    const ctx = canvas.getContext('2d');
    
    // We'll normalize the height to the first image's height for a clean side-by-side look
    const h1 = images.img1.height;
    const w1 = images.img1.width;
    
    const h2 = images.img2.height;
    const w2 = images.img2.width;
    
    // Scale second image to match first image's height
    const scaleFactor = h1 / h2;
    const scaledW2 = w2 * scaleFactor;
    
    canvas.width = w1 + scaledW2;
    canvas.height = h1;
    
    // Draw first image
    ctx.drawImage(images.img1, 0, 0);
    
    // Draw second image
    ctx.drawImage(images.img2, w1, 0, scaledW2, h1);
    
    previewContainer.classList.add('visible');
    previewContainer.scrollIntoView({ behavior: 'smooth' });
});

// Reset Logic
resetBtn.addEventListener('click', () => {
    images = { img1: null, img2: null };
    img1.src = '';
    img2.src = '';
    dropZone1.classList.remove('has-image');
    dropZone2.classList.remove('has-image');
    previewContainer.classList.remove('visible');
    fileInput1.value = '';
    fileInput2.value = '';
});

// Download Logic
downloadBtn.addEventListener('click', () => {
    const link = document.createElement('a');
    link.download = 'merged-image.png';
    link.href = canvas.toDataURL('image/png');
    link.click();
});
