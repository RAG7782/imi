#!/usr/bin/env node
/**
 * NotebookLM Automated Uploader
 *
 * Creates a Google NotebookLM notebook and adds files as text sources.
 * Uses the notebooklm CLI's auth (storage-state.json).
 *
 * Usage:
 *   node notebooklm_upload.mjs "<title>" file1.txt file2.md file3.py ...
 *   node notebooklm_upload.mjs --status                    # Check auth
 *   node notebooklm_upload.mjs --add <notebookId> file...  # Add to existing
 *
 * Notes:
 *   - Auth is managed by `notebooklm login` (run once, lasts ~7 days)
 *   - The CLI's add-text returns empty IDs but sources ARE added (known bug)
 *   - This script uses the CLI directly — no custom RPC needed
 */
import { execSync } from 'child_process';
import { readFileSync, writeFileSync } from 'fs';
import { basename } from 'path';

const DELAY_MS = 2000;
const sleep = ms => new Promise(r => setTimeout(r, ms));

function run(cmd, opts = {}) {
  try {
    return execSync(cmd, { encoding: 'utf8', timeout: 60000, ...opts }).trim();
  } catch(e) {
    return e.stderr || e.stdout || e.message;
  }
}

function checkAuth() {
  const result = run('notebooklm list 2>&1 | head -5');
  if (result.includes('Found') && result.includes('notebook')) {
    return true;
  }
  console.error('Auth check failed. Run: notebooklm login');
  return false;
}

function createNotebook(title) {
  console.log(`Creating notebook: "${title}"...`);
  const output = run(`notebooklm create "${title.replace(/"/g, '\\"')}" 2>&1`);

  // Extract ID from output
  const idMatch = output.match(/ID:\s*([0-9a-f-]{36})/i);
  if (idMatch) {
    console.log(`  ID: ${idMatch[1]}`);
    return idMatch[1];
  }

  // Fallback: list and find by title
  const listOutput = run('notebooklm list 2>&1');
  const lines = listOutput.split('\n');
  for (const line of lines) {
    if (line.includes(title.substring(0, 30))) {
      const match = line.match(/^([0-9a-f]{8})/);
      if (match) {
        console.log(`  ID (partial): ${match[1]}`);
        return match[1];
      }
    }
  }

  console.error('  Failed to create notebook');
  console.error('  Output:', output.substring(0, 300));
  return null;
}

function addSource(notebookId, filePath) {
  const title = basename(filePath).replace(/\.(txt|md|py|js|ts|json|yaml|yml)$/i, '');
  const content = readFileSync(filePath, 'utf8');

  console.log(`  [+] "${title}" (${content.length} chars)`);

  // Use add-text with --file (sources ARE added despite empty ID in output)
  const output = run(
    `notebooklm source add-text "${notebookId}" "${title.replace(/"/g, '\\"')}" --file "${filePath}" 2>&1`
  );

  const success = output.includes('Added source');
  if (success) {
    console.log(`      Added`);
  } else {
    console.log(`      Warning: ${output.substring(0, 100)}`);
  }
  return success;
}

async function main() {
  const args = process.argv.slice(2);

  if (args.includes('--status')) {
    if (checkAuth()) {
      console.log('Auth OK. Ready to create notebooks.');
    }
    return;
  }

  if (args.length < 1 || args[0] === '--help') {
    console.log(`
NotebookLM Uploader
====================
Usage:
  node notebooklm_upload.mjs "<title>" file1.txt file2.md ...
  node notebooklm_upload.mjs --add <notebookId> file1.txt ...
  node notebooklm_upload.mjs --status

Prerequisites:
  npm install -g notebooklm
  notebooklm login  (once, lasts ~7 days)
`);
    return;
  }

  // Check auth
  if (!checkAuth()) return;

  // Handle --add to existing notebook
  let notebookId;
  let files;

  if (args[0] === '--add') {
    notebookId = args[1];
    files = args.slice(2);
    console.log(`Adding to existing notebook: ${notebookId}\n`);
  } else {
    const title = args[0];
    files = args.slice(1);

    console.log('=== NotebookLM Uploader ===\n');
    notebookId = createNotebook(title);
    if (!notebookId) process.exit(1);
  }

  if (files.length === 0) {
    console.log('\nNo files specified. Notebook created empty.');
    return;
  }

  console.log(`\nAdding ${files.length} sources...\n`);

  let success = 0;
  let failed = [];

  for (let i = 0; i < files.length; i++) {
    try {
      const ok = addSource(notebookId, files[i]);
      if (ok) success++;
      else failed.push(files[i]);
    } catch(e) {
      console.error(`      Error: ${e.message}`);
      failed.push(files[i]);
    }

    if (i < files.length - 1) await sleep(DELAY_MS);
  }

  // Summary
  console.log(`\n${'='.repeat(50)}`);
  console.log(`Done: ${success}/${files.length} sources added`);
  if (failed.length > 0) console.log(`Failed: ${failed.join(', ')}`);
  console.log(`Notebook: https://notebooklm.google.com/notebook/${notebookId}`);
  console.log('='.repeat(50));

  // Save metadata
  const meta = {
    notebookId,
    title: args[0] === '--add' ? '(existing)' : args[0],
    sources: files.length,
    success,
    failed,
    createdAt: new Date().toISOString(),
    url: `https://notebooklm.google.com/notebook/${notebookId}`
  };
  writeFileSync('/tmp/notebooklm_last_upload.json', JSON.stringify(meta, null, 2));
}

main().catch(e => { console.error('Fatal:', e.message); process.exit(1); });
