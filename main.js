const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

console.log(`Current working directory: ${process.cwd()}`);
console.log(`App path: ${app.getAppPath()}`);

let mainWindow;

// Configuration
const CONFIG = {
    window: {
        width: 900,
        height: 700,
        minWidth: 800,
        minHeight: 600
    },
    backend: {
        executable: 'invoice-backend.exe',
        timeout: 300000 // 5 minutes
    },
    paths: {
        backend: path.join(__dirname, 'backend'),
        config: path.join(__dirname, 'config'),
        logs: path.join(__dirname, 'logs'),
        temp: path.join(__dirname, 'temp')
    }
};

// Ensure required directories exist
function ensureDirectories() {
    const dirs = [CONFIG.paths.logs, CONFIG.paths.temp];
    dirs.forEach(dir => {
        if (!fs.existsSync(dir)) {
            fs.mkdirSync(dir, { recursive: true });
            console.log(`Created directory: ${dir}`);
        }
    });
}

// Create the main application window
function createMainWindow() {
    mainWindow = new BrowserWindow({
        width: CONFIG.window.width,
        height: CONFIG.window.height,
        minWidth: CONFIG.window.minWidth,
        minHeight: CONFIG.window.minHeight,
        webPreferences: {
            nodeIntegration: true,
            contextIsolation: false,
            enableRemoteModule: false
        },
        icon: path.join(__dirname, 'assets', 'icon.png'), // Add your icon
        title: 'Invoice Data Extractor',
        backgroundColor: '#f5f5f5',
        show: false // Don't show until ready
    });

    // Load the frontend
    mainWindow.loadFile(path.join(__dirname, 'frontend', 'index.html'));

    // Show window when ready
    mainWindow.once('ready-to-show', () => {
        mainWindow.show();
        console.log('Window ready and shown');
    });

    // Handle window close
    mainWindow.on('closed', () => {
        mainWindow = null;
    });

    // Open DevTools in development mode
    if (process.env.NODE_ENV === 'development') {
        mainWindow.webContents.openDevTools();
    }
}

// IPC Handler: Select output folder
ipcMain.handle('select-output-folder', async () => {
    const result = await dialog.showOpenDialog(mainWindow, {
        properties: ['openDirectory'],
        title: 'Select Output Folder',
        buttonLabel: 'Select Folder'
    });

    if (result.canceled) {
        return null;
    }

    return result.filePaths[0];
});

// IPC Handler: Process PDFs
ipcMain.on('process-pdfs', (event, { filePaths, apiKey, outputFolder, filename }) => {
    console.log('='.repeat(60));
    console.log('Processing Request Received');
    console.log('='.repeat(60));
    console.log(`Files: ${filePaths.length}`);
    console.log(`Output: ${path.join(outputFolder, filename)}`);
    console.log(`API Key: ${apiKey ? '***' + apiKey.slice(-4) : 'Not provided'}`);

    // Validate inputs
    if (!filePaths || filePaths.length === 0) {
        event.reply('processing-result', {
            success: false,
            error: 'No files selected'
        });
        return;
    }

    if (!apiKey) {
        event.reply('processing-result', {
            success: false,
            error: 'API key is required'
        });
        return;
    }

    if (!outputFolder) {
        event.reply('processing-result', {
            success: false,
            error: 'Output folder is required'
        });
        return;
    }

    // Validate files exist
    const invalidFiles = filePaths.filter(file => !fs.existsSync(file));
    if (invalidFiles.length > 0) {
        event.reply('processing-result', {
            success: false,
            error: `Files not found: ${invalidFiles.join(', ')}`
        });
        return;
    }

    // Path to the backend executable
    const backendPath = path.join(CONFIG.paths.backend, CONFIG.backend.executable);
    console.log(`Backend path: ${backendPath}`);

    // Check if backend exists
    if (!fs.existsSync(backendPath)) {
        console.error('Backend executable not found!');
        event.reply('processing-result', {
            success: false,
            error: `Backend not found at: ${backendPath}\nPlease build the backend first.`
        });
        return;
    }

    // Prepare arguments for backend
    const args = [...filePaths, apiKey, outputFolder, filename];
    console.log(`Spawning backend with ${args.length} arguments`);

    // Spawn the backend process
    const backendProcess = spawn(backendPath, args, {
        cwd: CONFIG.paths.backend,
        windowsHide: true
    });

    let outputData = '';
    let errorData = '';
    let hasTimedOut = false;

    // Set timeout
    const timeout = setTimeout(() => {
        hasTimedOut = true;
        backendProcess.kill();
        console.error('Backend process timed out');
        event.reply('processing-result', {
            success: false,
            error: 'Processing timed out. Please try with fewer files or check your internet connection.'
        });
    }, CONFIG.backend.timeout);

    // Capture stdout
    backendProcess.stdout.on('data', (data) => {
        const message = data.toString();
        outputData += message;
        console.log(`[Backend] ${message.trim()}`);
        
        // Send progress updates to frontend
        event.reply('processing-progress', {
            message: message.trim()
        });
    });

    // Capture stderr
    backendProcess.stderr.on('data', (data) => {
        const message = data.toString();
        errorData += message;
        console.error(`[Backend Error] ${message.trim()}`);
    });

    // Handle process completion
    backendProcess.on('close', (code) => {
        clearTimeout(timeout);

        if (hasTimedOut) {
            return; // Already handled
        }

        console.log('='.repeat(60));
        console.log(`Backend process exited with code ${code}`);
        console.log('='.repeat(60));

        if (code === 0) {
            // Success
            const outputPath = path.join(outputFolder, filename);
            console.log(`✓ Processing successful`);
            console.log(`✓ Output saved to: ${outputPath}`);

            event.reply('processing-result', {
                success: true,
                output: outputPath,
                message: `Successfully processed ${filePaths.length} file(s)`
            });
        } else {
            // Error
            console.error(`✗ Processing failed with code ${code}`);
            console.error(`Error output: ${errorData}`);

            event.reply('processing-result', {
                success: false,
                error: errorData || `Process exited with code ${code}`,
                details: outputData
            });
        }
    });

    // Handle process errors
    backendProcess.on('error', (error) => {
        clearTimeout(timeout);
        console.error('Failed to start backend process:', error);
        
        event.reply('processing-result', {
            success: false,
            error: `Failed to start backend: ${error.message}`
        });
    });
});

// IPC Handler: Get application info
ipcMain.handle('get-app-info', async () => {
    return {
        name: app.getName(),
        version: app.getVersion(),
        paths: CONFIG.paths,
        platform: process.platform,
        arch: process.arch
    };
});

// IPC Handler: Open output folder
ipcMain.handle('open-output-folder', async (event, folderPath) => {
    const { shell } = require('electron');
    
    if (fs.existsSync(folderPath)) {
        await shell.openPath(folderPath);
        return { success: true };
    } else {
        return {
            success: false,
            error: 'Folder not found'
        };
    }
});

// IPC Handler: Get logs
ipcMain.handle('get-logs', async () => {
    try {
        const logsPath = CONFIG.paths.logs;
        if (!fs.existsSync(logsPath)) {
            return { logs: [], error: 'Logs directory not found' };
        }

        const logFiles = fs.readdirSync(logsPath)
            .filter(file => file.endsWith('.log'))
            .map(file => ({
                name: file,
                path: path.join(logsPath, file),
                size: fs.statSync(path.join(logsPath, file)).size,
                modified: fs.statSync(path.join(logsPath, file)).mtime
            }))
            .sort((a, b) => b.modified - a.modified);

        return { logs: logFiles };
    } catch (error) {
        return { logs: [], error: error.message };
    }
});

// IPC Handler: Clear cache
ipcMain.handle('clear-cache', async () => {
    try {
        const tempPath = CONFIG.paths.temp;
        if (fs.existsSync(tempPath)) {
            const files = fs.readdirSync(tempPath);
            files.forEach(file => {
                fs.unlinkSync(path.join(tempPath, file));
            });
        }
        return { success: true, message: 'Cache cleared successfully' };
    } catch (error) {
        return { success: false, error: error.message };
    }
});

// App lifecycle
app.whenReady().then(() => {
    console.log('='.repeat(60));
    console.log('Invoice Data Extractor Starting...');
    console.log('='.repeat(60));
    console.log(`Version: ${app.getVersion()}`);
    console.log(`Platform: ${process.platform}`);
    console.log(`Node: ${process.versions.node}`);
    console.log(`Electron: ${process.versions.electron}`);
    console.log('='.repeat(60));

    // Ensure directories exist
    ensureDirectories();

    // Create main window
    createMainWindow();

    // macOS: Re-create window when dock icon is clicked
    app.on('activate', () => {
        if (BrowserWindow.getAllWindows().length === 0) {
            createMainWindow();
        }
    });
});

// Quit when all windows are closed (except on macOS)
app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
        app.quit();
    }
});

// Handle app quit
app.on('before-quit', () => {
    console.log('Application quitting...');
    // Cleanup if needed
});

// Handle uncaught exceptions
process.on('uncaughtException', (error) => {
    console.error('Uncaught Exception:', error);
    
    if (mainWindow && !mainWindow.isDestroyed()) {
        mainWindow.webContents.send('app-error', {
            type: 'uncaught-exception',
            message: error.message,
            stack: error.stack
        });
    }
});

// Handle unhandled promise rejections
process.on('unhandledRejection', (reason, promise) => {
    console.error('Unhandled Rejection at:', promise, 'reason:', reason);
    
    if (mainWindow && !mainWindow.isDestroyed()) {
        mainWindow.webContents.send('app-error', {
            type: 'unhandled-rejection',
            message: String(reason)
        });
    }
});

console.log('Main process initialized');