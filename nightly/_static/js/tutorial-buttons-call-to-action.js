// This code replaces the default sphinx gallery download buttons
// with the 3 download buttons at the top of the page

document.addEventListener('DOMContentLoaded', function() {
  var downloadNote = $(".sphx-glr-download-link-note.admonition.note");
  if (downloadNote.length >= 1) {
    var tutorialUrlArray = $("#tutorial-type").text().split('/');
    tutorialUrlArray[0] = tutorialUrlArray[0] + "_source";

    var githubLink = "https://github.com/pytorch/tutorials/blob/main/" + tutorialUrlArray.join("/") + ".py";

    // Find the notebook download link - in pydata sphinx theme the structure might be different
    var notebookLinks = $(".reference.download");
    var notebookLink = notebookLinks.length > 1 ? notebookLinks[1].href : notebookLinks[0].href;

    var notebookDownloadPath = notebookLink.split('_downloads')[1];
    var colabLink = "https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads" + notebookDownloadPath;

    // Create the links directly instead of wrapping existing elements
    var colabElement = $("#google-colab-link");
    if (colabElement.length) {
      colabElement.find('div').wrapInner('<a href="' + colabLink + '" data-behavior="call-to-action-event" data-response="Run in Google Colab" target="_blank"></a>');
    }

    var notebookElement = $("#download-notebook-link");
    if (notebookElement.length) {
      notebookElement.find('div').wrapInner('<a href="' + notebookLink + '" data-behavior="call-to-action-event" data-response="Download Notebook"></a>');
    }

    var githubElement = $("#github-view-link");
    if (githubElement.length) {
      githubElement.find('div').wrapInner('<a href="' + githubLink + '" data-behavior="call-to-action-event" data-response="View on Github" target="_blank"></a>');
    }
    $(".sphx-glr-footer").hide();
    $(".sphx-glr-signature").hide();
  } else {
    $(".pytorch-call-to-action-links").hide();
  }
});
