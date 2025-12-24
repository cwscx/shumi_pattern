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
            <label>类型</label><select id="subType"><option>配方奶</option><option>母乳</option></select>
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
    html += `<button style="margin-top:10px; width:100%; background:#4caf50; color:white;" onclick="saveEntry('${actionType}')">确认保存</button>`;
    container.innerHTML = html;
}

async function saveEntry(action) {
    const date = document.getElementById('currentDate').value.replaceAll('-', '/');
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
    if (!confirm("确定要删除这条记录吗？")) return;

    // If dateStr isn't passed from the template, grab it from the date picker
    const date = dateStr || document.getElementById('currentDate').value.replaceAll('-', '/');

    try {
        const response = await fetch('/delete-action/', {
            method: 'POST',
            headers: { 
                'Content-Type': 'application/json',
                'X-CSRFToken': getCookie('csrftoken')
            },
            body: JSON.stringify({ date: date, index: index })
        });

        if (response.ok) {
            window.location.reload(); // Refresh to show updated list
        }
    } catch (error) {
        console.error('Error:', error);
    }
}

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