document.getElementById("summaryForm").addEventListener("submit", function(e) {

    e.preventDefault();

    const loader = document.getElementById("loader");
    const results = document.getElementById("results");

    loader.style.display = "block";
    results.style.display = "none";

    const formData = new FormData(this);

    fetch("/", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {

        document.getElementById("decisionResult").textContent = data.decision;

        loader.style.display = "none";
        results.style.display = "block";
    })
    .catch(error => {
        loader.style.display = "none";
        alert("Error generating summary.");
        console.error(error);
    });

});