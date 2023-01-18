function formSubmit(event) {
  document.getElementById("loadingSpinner").classList.remove("d-none");
  var url = "http://127.0.0.1:3000/predictions/video-captioning";
  var request = new XMLHttpRequest();
  request.open('POST', url, true);
  request.onreadystatechange = function() {
    if (this.readyState == 4) {
      if (this.status == 200) { // success
        document.getElementById('generatedCaption').innerHTML = request.responseText;
        document.getElementById("myData").value
      }
      document.getElementById("myData").value = "";
      document.getElementById("loadingSpinner").classList.add("d-none");
    }
  };

  request.send(new FormData(event.target)); // create FormData from form that triggered event
  event.preventDefault();
}

function fileInputChange(event){
  console.log(event.target.files[0])
  url = URL.createObjectURL(event.target.files[0]);
  document.querySelector("video").src = url

}

function attachFormSubmitEvent(formId){
  document.getElementById(formId).addEventListener("submit", formSubmit);
}
function attachFileInputChangeEvent(fileInputId){
  document.getElementById(fileInputId).addEventListener("change", fileInputChange);
}
attachFormSubmitEvent("formId")
attachFileInputChangeEvent("myData")