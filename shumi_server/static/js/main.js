// Orchestrates everything in index page and handles DOMContentLoaded
import { API } from './api.js';
import { UI } from './ui.js';
import { setCaliDefaultDate } from './utils.js';
import { initChart } from './chart_logic.js';

document.addEventListener('DOMContentLoaded', () => {
    setCaliDefaultDate();
    initChart();
});

// Expose functions to HTML onclick
window.addAction = UI.renderActionForm;

window.handleSave = async (action) => {
    const payload = {
        date: document.getElementById('currentDate').value.replaceAll('-', '/'),
        timezone: document.getElementById('timezoneSelect').value,
        action_item: {
            action,
            time_start: document.getElementById('timeStart').value,
            type: document.getElementById('subType')?.value,
            volume: document.getElementById('volume')?.value,
            time_end: document.getElementById('timeEnd')?.value
        }
    };
    const res = await API.saveEntry(payload);
    if (res.ok) window.location.reload();
};

window.deleteAction = async (dateStr, index) => {
    if (!confirm(`确定要删除 ${dateStr} 的这条记录吗？`)) return;

    try {
        const response = await API.deleteAction(dateStr, index);

        if (response.ok) {
            // Success! Reload to refresh the list and the chart
            window.location.reload();
        } else {
            const errorData = await response.json();
            alert("删除失败: " + (errorData.message || "未知错误"));
        }
    } catch (error) {
        console.error('Delete Error:', error);
        alert("网络连接失败，请重试");
    }
};

window.updatePrediction = async () => {
    const btn = document.getElementById('forecastBtn');
    const useLocal = document.getElementById('localModelToggle').checked;
    
    btn.disabled = true;
    document.getElementById('shumiLoader').style.display = 'flex';
    document.getElementById('nextAction').style.display = 'none';

    try {
        const response = await API.getPrediction(useLocal);
        const data = await response.json();
        if (data.status === 'success') {
            UI.updatePredictionDisplay(data);
        }
    } finally {
        document.getElementById('shumiLoader').style.display = 'none';
        btn.disabled = false;
    }
};

window.generateInsights = async () => {
    const query = document.getElementById('userQuery').value || "";
    const aiBtn = document.getElementById('aiBtn');
    const aiText = document.getElementById('aiText');
    const aiContent = document.getElementById('aiContent');

    // 1. UI Loading State
    aiBtn.disabled = true;
    aiBtn.innerText = "⏳ 正在运行分析...";
    aiContent.style.display = "block";
    aiText.innerText = "正在同步施舒米的最新数据...";

    try {
        const response = await API.getInsights(query);
        const data = await response.json();

        if (data.status === 'success') {
            // 2. Simple Formatting Logic
            // Converts \n to <br>, **bold** to <strong>, and * to bullet points
            aiText.innerHTML = data.insights
                .replace(/\n/g, '<br>')
                .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                .replace(/\* /g, '• ');
        } else {
            aiText.innerText = "Error: " + data.message;
        }
    } catch (e) {
        aiText.innerText = "无法连接到 AI 服务器，请检查网络。";
    } finally {
        // 3. Reset Button
        aiBtn.disabled = false;
        aiBtn.innerText = "🚀 重新运行分析";
    }
};

window.toggleReasoning = () => {
    const sec = document.getElementById('reasoningSection');
    sec.style.display = sec.style.display === "none" ? "block" : "none";
};