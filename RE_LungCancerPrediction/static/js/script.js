document.getElementById('uploadForm').addEventListener('submit', async function (e) {
    e.preventDefault();

    const fileInput = document.getElementById('imageInput');
    const file = fileInput.files[0];
    const resultElement = document.getElementById('result');
    const severityElement = document.getElementById('severity');
    const treatmentElement = document.getElementById('treatment');
    const predictButton = document.getElementById('predictBtn');

    if (!file) {
        resultElement.innerText = 'Please upload an image.';
        resultElement.className = 'error';
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }

        const data = await response.json();

        if (data.error) {
            resultElement.innerText = `Error: ${data.error}`;
            resultElement.className = 'error';
        } else {
            // Insert the prediction result into the HTML
            resultElement.innerHTML = `<strong>Prediction:</strong> ${data.prediction}`;

            // Insert the severity and treatment details into the HTML
            severityElement.innerHTML = `<strong>Severity:</strong> ${data.severity}`;
            treatmentElement.innerHTML = `<strong>Treatment:</strong> ${data.treatment}`;
        }

        // Show the prediction button after the image is uploaded
        predictButton.style.display = 'block';

    } catch (error) {
        resultElement.innerText = `Error: ${error.message}`;
        resultElement.className = 'error';
    }
});

// Show uploaded image preview
document.getElementById('imageInput').addEventListener('change', function (event) {
    const file = event.target.files[0];
    const uploadedImage = document.getElementById('uploadedImage');
    const predictButton = document.querySelector('.predict-btn');

    if (file) {
        const reader = new FileReader();
        reader.onload = function (e) {
            uploadedImage.src = e.target.result;
            uploadedImage.style.display = 'block';
        };
        reader.readAsDataURL(file);

        // Show the predict button after the image is uploaded
        predictButton.style.display = 'block';
    } else {
        uploadedImage.style.display = 'none';
        predictButton.style.display = 'none';
    }
});
