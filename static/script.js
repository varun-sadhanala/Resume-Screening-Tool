document.addEventListener("DOMContentLoaded", function() {
    // Automatically hide flash messages after 5 seconds
    setTimeout(function() {
        const flashMessages = document.querySelector('.flash-messages');
        if (flashMessages) {
            flashMessages.style.display = 'none';
        }
    }, 5000); // Adjust time as needed
});
