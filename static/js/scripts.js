document.addEventListener("DOMContentLoaded", () => {
    const input = document.getElementById("image-input");
    const preview = document.getElementById("preview");

    if (!input || !preview) {
        return;
    }

    input.addEventListener("change", () => {
        const file = input.files && input.files[0];
        if (!file) {
            preview.classList.add("d-none");
            preview.removeAttribute("src");
            return;
        }
        preview.src = URL.createObjectURL(file);
        preview.classList.remove("d-none");
    });
});

