/* Manual end-to-end UI review helper, not a unit-test suite.
   Requires Playwright and Chromium (or CHROME_BINARY).
   Start the Streamlit frontend before running this script. */
const { chromium } = require('playwright');
const fs = require('node:fs');
const path = require('node:path');

(async () => {
  const output = path.resolve(process.env.SCREENSHOT_DIR || 'artifacts/screenshots');
  fs.mkdirSync(output, {recursive: true});
  const browser = await chromium.launch({
    headless: true,
    ...(process.env.CHROME_BINARY ? {executablePath: process.env.CHROME_BINARY} : {})
  });
  const page = await browser.newPage({viewport: {width: 1200, height: 1100}, deviceScaleFactor: 1});
  const errors = [];
  async function stableView() {
    await page.waitForFunction(() => !document.querySelector('[data-stale="true"]'), {timeout: 30000});
    await page.evaluate(() => {
      window.scrollTo(0, 0);
      document.querySelectorAll('*').forEach(element => { if (element.scrollTop) element.scrollTop = 0; });
    });
    await page.waitForTimeout(2500);
  }
  page.on('pageerror', error => errors.push(error.message));
  await page.goto(process.env.FRONTEND_URL || 'http://localhost:8501', {waitUntil: 'domcontentloaded'});
  await page.getByText('Completed evaluations', {exact: true}).waitFor({timeout: 90000});
  await page.getByText('Experiment protocol', {exact: true}).waitFor({timeout: 90000});
  await page.getByRole('button', {name: 'Stop', exact: true}).waitFor({state: 'hidden', timeout: 90000});
  await page.waitForTimeout(1200);
  await stableView();
  await page.screenshot({path: path.join(output, '01_overview.png')});
  await page.getByText('Experiment protocol', {exact: true}).scrollIntoViewIfNeeded();
  await page.screenshot({path: path.join(output, '07_overview_charts.png')});
  for (const [view, file, ready] of [
    ['Model comparison', '02_model_comparison.png', 'Computing cost'],
    ['Prediction explorer', '03_prediction_explorer.png', 'Actual positives'],
    ['Dataset details', '04_dataset_details.png', 'Download dataset manifest'],
    ['Reproducibility', '05_reproducibility.png', 'Reproduce this experiment']
  ]) {
    await page.getByText(view, {exact: true}).click();
    await page.getByText(ready, {exact: true}).waitFor({timeout: 30000});
    await page.getByRole('button', {name: 'Stop', exact: true}).waitFor({state: 'hidden', timeout: 90000});
    if (await page.locator('[data-testid="stException"]').count()) {
      throw new Error(await page.locator('[data-testid="stException"]').innerText());
    }
    await page.waitForTimeout(1200);
    await stableView();
    await page.screenshot({path: path.join(output, file)});
  }
  await page.getByText('Model comparison', {exact: true}).click();
  const downloadPromise = page.waitForEvent('download');
  await page.getByRole('button', {name: 'Download model results'}).click();
  const download = await downloadPromise;
  await download.saveAs(path.join(output, 'download_check.csv'));
  await page.getByRole('combobox').nth(1).click();
  await page.getByRole('option', {name: 'SMD machine-2-1', exact: true}).click();
  await page.getByRole('heading', {name: 'SMD machine-2-1', exact: true}).waitFor();
  await page.getByRole('button', {name: 'Stop', exact: true}).waitFor({state: 'hidden', timeout: 90000});
  await page.waitForTimeout(1200);
  await stableView();
  await page.screenshot({path: path.join(output, '06_smd_comparison.png')});
  fs.writeFileSync(path.join(output, 'ui_review.json'), JSON.stringify({
    checked_at: new Date().toISOString(), views: 5, download: download.suggestedFilename(),
    browser_errors: errors
  }, null, 2));
  await browser.close();
  if (errors.length) throw new Error(errors.join('\n'));
  console.log('Reviewed five views and verified a CSV download. Screenshots saved.');
})().catch(error => {console.error(error); process.exit(1);});
