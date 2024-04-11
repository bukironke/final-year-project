const questions = [
    { question: 'What\'s your name?', validation: /^[a-zA-Z]+$/},
    { question: 'How old are you?', validation: /^[0-9]+$/},
    { question: 'What is your annual salary? No commas.', validation: /^[0-9]+$/},
    { question: 'Do you have a mortgage, renting or own a property?', validation: /^(mortgage|rent|own)$/i},
    { question: 'Have you defaulted on a loan in the last 3 months?', validation: /^(yes|no)$/i},
    { question: 'How many years have you been employed for?', validation: /^[0-9]+$/},
    { question: 'How many years have you been on the credit bureau for?', validation: /^[0-9]+$/},
];

const shakeTime= 100;
const switchTime = 200;


let position = 0;


const formBox = document.querySelector('#form-box');
const prevBtn = document.querySelector('#prev-btn');
const nextBtn = document.querySelector('#next-btn');
const inputGroup = document.querySelector('#input-group');
const inputField = document.querySelector('#input-field');
const inputLabel = document.querySelector('#input-label');
const inputProgress = document.querySelector('#input-progress');
const progress = document.querySelector('#progress-bar');

document.addEventListener('DOMContentLoaded', getQuestion);

nextBtn.addEventListener('click', validate);

inputField.addEventListener('keyup', e => {
    if (e.keyCode == 13) {
        validate();
    }
});

prevBtn.addEventListener("click", () => {
    position = position - 1;
    getQuestion();
  });


//Functions

function getQuestion() {
    inputLabel.innerHTML = questions[position].question;

    inputField.type = questions[position].type || 'text';

    inputField.value = questions[position].answer || '';

    inputField.focus();

    progress.style.width = (position * 100) / questions.length + '%';

    prevBtn.className = position ? 'lni-arrow-left' : 'lni-user';

    showQuestion();
}

function showQuestion() {
    inputGroup.style.opacity = 1;
    inputProgress.style.transition = '';
    inputProgress.style.width= '100%'
}

function hideQuestion() {
    inputGroup.style.opacity = 0;
    inputProgress.style.transition = 'none';
    inputProgress.style.width = 0;
    inputLabel.style.marginLeft = 0;
    inputGroup.style.border = null;
}

function transform(x, y) {
    formBox.style.transform = `translate(${x}px, ${y}px)`;
}

function validate() {
    const inputValue = inputField.value.trim();
    const validationRegex = questions[position].validation;

    if (!validationRegex.test(inputValue)) {
        inputFail();
    } else {
        inputPass();
    }
}

function inputFail() {
    formBox.className = 'error';

    for (let i = 0; i < 6; i++ ) {
        setTimeout(transform, shakeTime * i, ((i % 2) * 2 - 1) * 20, 0);

        setTimeout(transform, shakeTime * 6, 0, 0);

        inputField.focus();
    }
}

function inputPass() {
    formBox.className = '';

    setTimeout(transform, shakeTime * 0, 0, 10);
    setTimeout(transform, shakeTime * 1, 0, 0);

    questions[position].answer = inputField.value;

    position++;

    if (questions[position]) {
        hideQuestion();
        getQuestion();
    } else {
        hideQuestion();
        formBox.className = 'close';
        progress.style.width = '100%';

        formComplete();
    }
}

function formComplete() {
    localStorage.setItem('userInputs', JSON.stringify(questions.map(q => q.answer)));
    const userInputs = JSON.parse(localStorage.getItem('userInputs'));
    const formData = {
        'Name': userInputs[0],
        'Age': parseInt(userInputs[1]),
        'Income': parseInt(userInputs[2]),
        'HomeOwnership': encodeHomeOwnership(userInputs[3]),
        'DefaultedLoan': userInputs[4].toLowerCase() === 'yes' ? 1 : 0,
        'EmploymentYears': parseInt(userInputs[5]),
        'CreditHistoryYears': parseInt(userInputs[6]),
    };

    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        window.location.href = '/results'; 
    })
    .catch(error => {
        console.error('Error:', error);
    });
}

function encodeHomeOwnership(homeOwnership) {
    switch(homeOwnership.toLowerCase()) {
        case 'mortgage': return 2;
        case 'rent': return 0;
        case 'own': return 1;
        default: return 0; // Handle unexpected values
    }
}

