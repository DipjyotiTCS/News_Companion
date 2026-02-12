const landingEl = document.getElementById("landing");
const threadEl = document.getElementById("thread");
const inputEl = document.getElementById("chatInput");
const sendBtn = document.getElementById("sendBtn");

function setSending(isSending){
    sendBtn.disabled = isSending;
    inputEl.disabled = isSending;
}

function showThread(){
    if (landingEl) landingEl.style.display = "none";
    if (threadEl) threadEl.style.display = "block";
}

function addMsg(role, text){
    const div = document.createElement("div");
    div.className = `wp-msg ${role === "user" ? "user" : "assistant"}`;
    div.textContent = text;
    threadEl.appendChild(div);
    threadEl.scrollTop = threadEl.scrollHeight;
}

async function loadHistory(){
    try{
        const res = await fetch("/api/history");
        const data = await res.json();
        const hist = data.history || [];
        if (hist.length > 0){
            showThread();
            hist.forEach(m => addMsg(m.role, m.content));
        }
    }catch(e){
        // ignore
    }
}

async function send(){
    const text = (inputEl.value || "").trim();
    if(!text) return;
    if (!threadEl) return;

    // If first message, switch from landing to thread.
    showThread();
    addMsg("user", text);
    inputEl.value = "";

    setSending(true);
    try{
        const res = await fetch("/api/chat", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({message: text})
        });
        const data = await res.json();
        if(!res.ok){
            addMsg("assistant", data.error || "Something went wrong.");
        }else{
            addMsg("assistant", data.reply || "");
        }
    }catch(e){
        addMsg("assistant", "Network error. Check the server logs and your connection.");
    }finally{
        setSending(false);
        inputEl.focus();
    }
}

// Tab UI (visual only)
document.querySelectorAll(".wp-tab").forEach(btn => {
    btn.addEventListener("click", () => {
        document.querySelectorAll(".wp-tab").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
    });
});

// Prompt buttons
document.querySelectorAll(".wp-prompt").forEach(btn => {
    btn.addEventListener("click", () => {
        const p = btn.getAttribute("data-prompt") || btn.textContent.trim();
        inputEl.value = p;
        inputEl.focus();
    });
});

sendBtn.addEventListener("click", send);
inputEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey){
        e.preventDefault();
        send();
    }
});

loadHistory();
inputEl.focus();
