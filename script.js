var userInput = document.getElementById("userInp");
var messages = [];
var regPattern = /<think>(.*?)<\/think>/i;
console.log("Hi I am Bixter ... !!");
var url = "http://localhost:11434";
const chatBox = document.getElementById("messages");
const models=[];
const modelSelect = document.getElementById("modelSelect");

const modelList = async () => {
    try {
        const res = await fetch(
            url+"/api/tags",
            {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json'
                }
            }
        );
        if (!res.ok) {
            throw Error("Response error");
        }
        const modelList = await res.json();
        console.log("Response received:", modelList);
        modelList["models"].forEach((model) => {
            models.push(model["model"]);
            const option = document.createElement("option");
            option.value = model["model"];
            option.textContent = model["name"];
            modelSelect.appendChild(option);
        });
    }
    catch (error) {
        console.error("Error occurred:", error);
    }
}
modelList();

const getResponse = async () => {
    try {
        const res = await fetch(
            url+"/api/chat",
            {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    "model": modelSelect.value,
                    "messages": [messages[messages.length - 1]],
                    "stream": false,
                })
            }
        );
        addLoading();
        if (!res.ok) {
            throw Error("Response error");
        }
        const botResponse = await res.json();
        console.log("Response received:", botResponse.message);
        addBotMsg(botResponse.message);
    }
    catch (error) {
        console.error("Error occurred:", error);
    }
};

userInput.addEventListener("keypress", (event) => {
    if (event.key === "Enter") {
        sendMsg();
    }
});

const clearChat = () => {
    messages = [];
    chatBox.innerHTML = "";
};

const getThinkMsg=(str)=>{
    const thinkMsg = {
        role: "assistant",
        content: str.replace(/<think>([\s\S]*?)<\/think>/i, "").trim()
    }
    return thinkMsg;
}
const addLoading=()=>{
    const loadingDiv = document.createElement("div");
    loadingDiv.className = "loading";
    loadingDiv.id="loading"
    loadingDiv.innerHTML = `<span class="dot"></span><span class="dot"></span><span class="dot"></span>`;
    chatBox.appendChild(loadingDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
}
const deleteLoading=()=>{
    const loadingElement = document.getElementById("loading");
    if (loadingElement) {
        loadingElement.remove();
    }
}


const addBotMsg = (msg) => {
    console.log("Messages array:", messages);
    var msg=getThinkMsg(msg["content"]);
    messages.push(msg);
    const msgDiv = document.createElement("div");
    msgDiv.className = msg["role"];
    msgDiv.innerHTML = `${marked.parse(msg['content'])}`;
    chatBox.appendChild(msgDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
};

const removeWelcm=()=>{
    const wlcm=document.getElementById("welcome-msg");
    wlcm.style="display:none;"
}


const addUsrMsg = (msg) => {
    const role = "user"; 
    messages.push({
        "role": role,
        "content": msg
    });
    if (messages.length>=1){
        removeWelcm();
    }
    console.log("Messages array:", messages);
    const msgDiv = document.createElement("div");
    msgDiv.className = role;
    msgDiv.innerHTML = `${msg}`;
    chatBox.appendChild(msgDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
};

const sendMsg = () => {
    if (userInput.value !== "") {
        addUsrMsg(userInput.value);
        userInput.value = "";
        getResponse();
    }
    else {
        alert("Prompt cannot be empty");
    }
};