const fs = require('fs');
const vm = require('vm');
const script = fs.readFileSync('debug_script.js', 'utf8');
const sandbox = {
  document: {
    getElementById(id) {
      return {
        appendChild() {},
        classList: { add() {}, remove() {} },
        style: {},
        innerHTML: '',
        textContent: '',
        addEventListener() {},
      };
    }
  },
  console: console,
  window: {},
};
try {
  vm.runInNewContext(script, sandbox);
  console.log('script executed');
} catch (err) {
  console.error('ERROR', err.name, err.message);
  console.error(err.stack.split('\n')[0]);
}
