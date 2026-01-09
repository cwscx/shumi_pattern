//This handles DOM manipulation, showing/hiding loaders, and rendering the dynamic form.import { getCurrentTime } from './utils.js';
import { getCurrentTime } from './utils.js';

export const UI = {
    renderActionForm: (actionType) => {
        const container = document.getElementById('dynamicInputs');
        container.style.display = 'block';
        
        // Add a class for styling the container
        container.className = 'dynamic-form-container active';

        let html = `<div class="form-header">
                        <span class="form-icon">${UI.getIcon(actionType)}</span>
                        <h4>记录 ${actionType}</h4>
                    </div>`;

        html += `<div class="form-body">`;

        if (actionType === '喝奶') {
            html += `
                <div class="form-group">
                    <label>奶类</label>
                    <select id="subType" class="styled-input"><option>配方奶</option><option>瓶喂母乳</option><option>亲喂母乳</option></select>
                </div>
                <div class="form-group">
                    <label>奶量 (ml)</label>
                    <input type="text" id="volume" class="styled-input" value="130ml">
                </div>`;
        } else if (actionType === '换尿布') {
            html += `
                <div class="form-group">
                    <label>类型</label>
                    <select id="subType" class="styled-input"><option>嘘嘘</option><option>臭臭</option><option>干爽</option></select>
                </div>`;
        } else if (actionType === '睡眠') {
            html += `
                <div class="form-group">
                    <label>结束时间 (可选)</label>
                    <input type="time" id="timeEnd" class="styled-input">
                </div>`;
        }

        html += `
            <div class="form-group">
                <label>开始时间</label>
                <input type="time" id="timeStart" class="styled-input" value="${getCurrentTime()}">
            </div>
        </div>`; // Close form-body

        html += `<button class="btn-confirm-save" onclick="handleSave('${actionType}')">确认保存到日志</button>`;
        
        container.innerHTML = html;
    },

    // Helper to get icons for the header
    getIcon: (type) => {
        const icons = { '喝奶': '🍼', '换尿布': '🧷', '睡眠': '😴' };
        return icons[type] || '📝';
    },

    updatePredictionDisplay: (data) => {
        const actionEl = document.getElementById('nextAction');
        const reasoningSec = document.getElementById('reasoningSection');
        const expandBtn = document.getElementById('expandBtn');
        const meterWrap = document.getElementById('confidenceWrapper');

        actionEl.innerText = data.prediction;
        reasoningSec.innerText = data.reasoning;
        actionEl.style.display = "block";
        expandBtn.style.display = "block";
        meterWrap.style.display = "block";
        
        const conf = parseInt(data.confidence);
        const meterFill = document.getElementById('meterFill');
        meterFill.style.width = conf + "%";
        document.getElementById('confValue').innerText = conf + "%";
        
        if (conf > 80) meterFill.style.backgroundColor = "#4caf50";
        else if (conf > 50) meterFill.style.backgroundColor = "#ffeb3b";
        else meterFill.style.backgroundColor = "#f44336";
    }
};