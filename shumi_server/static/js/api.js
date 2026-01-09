// static/js/api.js
import { getCookie } from './utils.js';

async function basePost(url, payload) {
    const response = await fetch(url, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': getCookie('csrftoken')
        },
        body: JSON.stringify(payload)
    });
    return response;
}

export const API = {
    saveEntry: (payload) => basePost('/save-baby-data/', payload),
    deleteAction: (date, index) => basePost('/delete-action/', { date, index }),
    getInsights: (query) => basePost('/get-insights/', { query }),
    getPrediction: (useLocalModel) => basePost('/get-prediction/', { useLocalModel })
};