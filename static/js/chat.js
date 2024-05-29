 //SweetAlert2 Firing modal
 const swalWithBootstrapButtons = Swal.mixin({
    customClass: {
    confirmButton: "btn btn-success",
    cancelButton: "btn btn-danger"
    },
    buttonsStyling: false
});
document.addEventListener('DOMContentLoaded', function() {
    function scrollToBottom() {
        console.log("Scroll to bottom function called");
        var chatBox = document.getElementById('chat-box');
        console.log("Chat box height:", chatBox.scrollHeight);
        chatBox.scrollTop = chatBox.scrollHeight;
    }

    function sendGPT(){
        $('#chat-box').append('<div class="row gpt_message_container">\
                                    <p class="gpt_message">MedGPT:<br>aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa</p>\
                                </div>'); 
        scrollToBottom();
    }
    
    function sendUserMessage(){
        var message = document.getElementById("message").value;
        if(message){
            $('#chat-box').append('<div class="row user_message_container">\
                                        <p class="user_message">User:<br>'+message+'</p>\
                                    </div>'); 
            document.getElementById("message").value = '';
            scrollToBottom();
            sendGPT();
        }
    }

    // Ensure functions are globally accessible
    window.sendGPT = sendGPT;
    window.sendUserMessage = sendUserMessage;

});

// function to toggle between light and dark theme
function toggleTheme() {
    const html = document.documentElement;
    const themeIcon = document.getElementById('themeIcon');
    if(html.getAttribute('data-bs-theme') === 'light') {
        html.setAttribute('data-bs-theme', 'dark');
        themeIcon.classList.remove('fa-sun');
        themeIcon.classList.add('fa-moon');
    } else {
        html.setAttribute('data-bs-theme', 'light');
        themeIcon.classList.remove('fa-moon');
        themeIcon.classList.add('fa-sun');
    }
}

function exit(){
    swalWithBootstrapButtons.fire({
        title: "Are you sure to quit the chat?",
      /*   text: "You won't be able to revert this!", */
        icon: "warning",
        showCancelButton: true,
        confirmButtonText: "Yes, I'm sure!",
        cancelButtonText: "No, bring me back!",
        text: "If you quit, you will not be able to recover the chat!",
        reverseButtons: true
      }).then((result) => {
        if (result.isConfirmed) {
            window.location.href = "/";
        }
      });
}