// 全局变量
let updateInterval;

// DOM 加载完成后执行
window.addEventListener('DOMContentLoaded', () => {
    // 启动状态更新
    startStatusUpdates();
    
    // 绑定事件
    document.getElementById('send-command').addEventListener('click', sendCommand);
    document.getElementById('command-input').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            sendCommand();
        }
    });
});

// 启动状态更新
function startStatusUpdates() {
    // 立即更新一次
    updateStatus();
    
    // 设置定时更新（100ms）
    updateInterval = setInterval(updateStatus, 100);
}

// 更新状态信息
async function updateStatus() {
    try {
        const response = await fetch('/api/status');
        if (!response.ok) {
            throw new Error('网络响应错误');
        }
        
        const data = await response.json();
        
        // 更新位置信息
        document.getElementById('pos-x').textContent = data.position.x.toFixed(2);
        document.getElementById('pos-y').textContent = data.position.y.toFixed(2);
        document.getElementById('pos-z').textContent = data.position.z.toFixed(2);
        
        // 更新姿态信息
        document.getElementById('ori-x').textContent = data.orientation.x.toFixed(2);
        document.getElementById('ori-y').textContent = data.orientation.y.toFixed(2);
        document.getElementById('ori-z').textContent = data.orientation.z.toFixed(2);
        document.getElementById('ori-w').textContent = data.orientation.w.toFixed(2);
        
        // 更新最后更新时间
        if (data.last_update_time) {
            document.getElementById('update-time').textContent = `最后更新: ${data.last_update_time}`;
        }
        
        // 更新状态显示
        const statusElement = document.getElementById('connection-status');
        statusElement.textContent = `状态: ${data.status}`;
        
        // 更新状态颜色
        if (data.status.includes('错误')) {
            statusElement.className = 'status error';
        } else if (data.status === '运行中') {
            statusElement.className = 'status running';
        } else {
            statusElement.className = 'status';
        }
        
        // 更新最新指令
        if (data.latest_command) {
            document.getElementById('latest-command-text').textContent = data.latest_command;
        }
        
    } catch (error) {
        console.error('更新状态时出错:', error);
        const statusElement = document.getElementById('connection-status');
        statusElement.textContent = '状态: 连接错误';
        statusElement.className = 'status error';
    }
}

// 发送指令
async function sendCommand() {
    const commandInput = document.getElementById('command-input');
    const command = commandInput.value.trim();
    
    if (!command) {
        alert('请输入指令');
        return;
    }
    
    try {
        const response = await fetch('/api/send_command', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ command: command })
        });
        
        if (!response.ok) {
            throw new Error('发送指令失败');
        }
        
        const result = await response.json();
        if (result.status === 'success') {
            // 清空输入框
            commandInput.value = '';
            console.log('指令发送成功:', result.received_command);
        } else {
            console.error('指令发送失败:', result.message);
        }
        
    } catch (error) {
        console.error('发送指令时出错:', error);
        alert('发送指令失败，请检查网络连接');
    }
}

// 页面卸载时清理定时器
window.addEventListener('beforeunload', () => {
    if (updateInterval) {
        clearInterval(updateInterval);
    }
});