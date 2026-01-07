// Set default date to today
document.addEventListener('DOMContentLoaded', () => {
    const today = new Date().toISOString().split('T')[0];
    document.getElementById('currentDate').value = today;
});

function getCurrentTime() {
    return new Date().toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' });
}

function addAction(actionType) {
    const container = document.getElementById('dynamicInputs');
    container.style.display = 'block';
    let html = `<h4>记录 ${actionType}</h4>`;

    if (actionType === '喝奶') {
        html += `
            <label>类型</label><select id="subType"><option>配方奶</option><option>瓶喂母乳</option><option>亲喂母乳</option></select>
            <label>奶量 (ml)</label><input type="text" id="volume" value="130ml">
        `;
    } else if (actionType === '换尿布') {
        html += `
            <label>类型</label><select id="subType"><option>嘘嘘</option><option>臭臭</option><option>干爽</option></select>
        `;
    } else if (actionType === '睡眠') {
        html += `<label>结束时间 (可选)</label><input type="time" id="timeEnd">`;
    }

    html += `<label>开始时间</label><input type="time" id="timeStart" value="${getCurrentTime()}">`;
    html += `<button style="margin-top:10px; width:100%; height: 40px; background:#4caf50; color:white;" onclick="saveEntry('${actionType}')">确认保存</button>`;
    container.innerHTML = html;
}

async function saveEntry(action) {
    const date = document.getElementById('currentDate').value.replaceAll('-', '/');
    const timezone = document.getElementById('timezoneSelect').value;
    const timeStart = document.getElementById('timeStart').value;
    
    let actionItem = { action, time_start: timeStart };

    if (action === '喝奶') {
        actionItem.type = document.getElementById('subType').value;
        actionItem.volume = document.getElementById('volume').value;
    } else if (action === '换尿布') {
        actionItem.type = document.getElementById('subType').value;
    } else if (action === '睡眠') {
        const end = document.getElementById('timeEnd').value;
        if (end) actionItem.time_end = end;
    }

    const payload = {
        date: date,
        timezone: timezone,
        action_item: actionItem
    };

    try {
        const response = await fetch('/save-baby-data/', {
            method: 'POST',
            headers: { 
                'Content-Type': 'application/json',
                'X-CSRFToken': getCookie('csrftoken') // Important for Django security
            },
            body: JSON.stringify(payload)
        });

        if (response.ok) {
            window.location.reload();
            // You could trigger a function here to refresh a history list
        }
    } catch (error) {
        console.error('Error:', error);
    }
}

async function deleteAction(dateStr, index) {
    if (!confirm(`确定要删除 ${dateStr} 的这条记录吗？`)) return;

    try {
        const response = await fetch('/delete-action/', {
            method: 'POST',
            headers: { 
                'Content-Type': 'application/json',
                'X-CSRFToken': getCookie('csrftoken')
            },
            body: JSON.stringify({ 
                date: dateStr, // Use the specific date from the group
                index: index 
            })
        });

        if (response.ok) {
            window.location.reload();
        }
    } catch (error) {
        console.error('Error:', error);
    }
}

async function generateInsights() {
    const query = document.getElementById('userQuery').value || "";
    const aiBtn = document.getElementById('aiBtn');
    const aiText = document.getElementById('aiText');
    const aiContent = document.getElementById('aiContent');

    aiBtn.disabled = true;
    aiBtn.innerText = "⏳ 正在运行分析...";
    aiContent.style.display = "block";

    try {
        const response = await fetch('/get-insights/', {
            method: 'POST',
            headers: { 
                'Content-Type': 'application/json',
                'X-CSRFToken': getCookie('csrftoken')
            },
            body: JSON.stringify({ query: query })
        });

        const data = await response.json();
        if (data.status === 'success') {
            // Simple hack to handle line breaks. 
            // For real markdown, you could use a library like 'marked.js'
            aiText.innerHTML = data.insights
                .replace(/\n/g, '<br>')
                .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                .replace(/\* /g, '• ');
        } else {
            aiText.innerText = "Error: " + data.message;
        }
    } catch (e) {
        aiText.innerText = "Failed to connect to AI server.";
    } finally {
        aiBtn.disabled = false;
        aiBtn.innerText = "🚀 重新运行分析";
    }
}

// Prediction
async function updatePrediction() {
    const actionEl = document.getElementById('nextAction');
    const loaderEl = document.getElementById('shumiLoader');
    const meterWrap = document.getElementById('confidenceWrapper');
    const forecastBtn = document.getElementById('forecastBtn');
    const meterFill = document.getElementById('meterFill');
    const confValue = document.getElementById('confValue');
    const reasoningSec = document.getElementById('reasoningSection');
    const expandBtn = document.getElementById('expandBtn');
    const useLocalModel = document.getElementById('localModelToggle').checked;
    
    // 1. Enter Loading State
    actionEl.style.display = "none";
    meterWrap.style.display = "none";
    loaderEl.style.display = "flex";
    forecastBtn.disabled = true;
    forecastBtn.innerText = "⚡ 正在同步信号...";
    reasoningSec.style.display = "none";

    try {
        const response = await fetch('/get-prediction/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': getCookie('csrftoken')
            },
            // Pass the toggle state to Python
            body: JSON.stringify({ 
                useLocalModel: useLocalModel 
            })
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            console.log(data);
            // 2. Hide Loader & Show Result
            loaderEl.style.display = "none";
            actionEl.style.display = "block";
            actionEl.innerText = data.prediction;
            reasoningSec.innerText = data.reasoning;
            expandBtn.style.display = "block";
            
            // 3. Animate Confidence Meter
            meterWrap.style.display = "block";
            setTimeout(() => {
                const conf = parseInt(data.confidence);
                meterFill.style.width = conf + "%";
                confValue.innerText = conf + "%";
                
                // Color Logic
                if (conf > 80) meterFill.style.backgroundColor = "#4caf50";
                else if (conf > 50) meterFill.style.backgroundColor = "#ffeb3b";
                else meterFill.style.backgroundColor = "#f44336";
            }, 100);
        }
    } catch (e) {
        loaderEl.style.display = "none";
        actionEl.style.display = "block";
        actionEl.innerText = "连接失败";
    } finally {
        forecastBtn.disabled = false;
        forecastBtn.innerText = "📡 再次同步预判";
    }
}

function toggleReasoning() {
    const sec = document.getElementById('reasoningSection');
    const btn = document.getElementById('expandBtn');
    if (sec.style.display === "none") {
        sec.style.display = "block";
        btn.innerText = "收起详情";
    } else {
        sec.style.display = "none";
        btn.innerText = "查看推理详情";
    }
}


// The chart logic
document.addEventListener('DOMContentLoaded', function() {
    const dataElement = document.getElementById('chart-data');
    if (!dataElement) return;
    const rawData = JSON.parse(dataElement.textContent);

    const ctx = document.getElementById('routineChart').getContext('2d');
    
    new Chart(ctx, {
        type: 'bar',
        data: {
            datasets: [
                {
                    label: '😴 睡眠',
                    data: rawData.sleep,
                    backgroundColor: 'rgba(156, 39, 176, 0.7)',
                    borderColor: '#9c27b0',
                    borderWidth: 1,
                    borderRadius: 4,
                    grouped: false // Keeps all blocks in the same column
                },
                {
                    label: '🍼 喝奶',
                    data: rawData.milk,
                    backgroundColor: 'rgba(33, 150, 243, 0.7)',
                    borderColor: '#2196f3',
                    borderWidth: 1,
                    borderRadius: 4,
                    grouped: false
                },
                {
                    label: '🧷 尿布',
                    data: rawData.diaper,
                    backgroundColor: 'rgba(255, 152, 0, 0.7)',
                    borderColor: '#ff9800',
                    borderWidth: 1,
                    borderRadius: 4,
                    grouped: false
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false, // Required to respect the container's height
            scales: {
                x: {
                    title: { display: true, text: '观察日期' },
                    grid: { display: false } // Cleans up the look
                },
                y: {
                    min: 0,
                    max: 24,
                    reverse: true, 
                    title: { display: true, text: '时间 (点击柱状块查看详情)' },
                    ticks: {
                        stepSize: 1, // Show a label for every single hour
                        autoSkip: false, // Force all hour labels to show
                        callback: value => {
                            // Formats 0-24 into 00:00 - 24:00
                            return value.toString().padStart(2, '0') + ":00";
                        },
                        font: {
                            size: 11
                        }
                    },
                    grid: {
                        color: '#ececec' // Light grid lines for every hour
                    }
                }
            },
            plugins: {
                legend: {
                    position: 'top',
                }
            }
        }
    });
});

// Helper function to get CSRF token from cookies
function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}