(function(Scratch) {
  'use strict';

  const vm = Scratch.vm
  const runtime = vm.runtime
  
  let jwArray = {
        Type: class { constructor(array) {/* noop */} static toArray(x) {/* noop */} },
        Block: {},
        Argument: {}
  }

  const u = x => { if (x instanceof jwArray.Type) x = x.toJSON(); return x; };

  function getTensorShape(a) {
    const s = [];
    for (; Array.isArray(a); a = a[0]) {
      const l = a.length;
      if (!l) return [0];
      s.push(l);
      const f = a[0], t = Array.isArray(f);
      for (let i = 1; i < l; i++) {
        const x = a[i];
        if (Array.isArray(x) !== t || (t && x.length !== f.length)) return [];
      }
    }
    return s;
  }

  function reshapeTensor(tensor, shape) {
    if (shape.some(el => typeof el !== 'number' || isNaN(el) || el <= 0)) return [];
    const flat = tensor.flat(Infinity);

    let size = 1;
    for (let i = 0; i < shape.length; i++) size *= shape[i];

    flat.length = size; // truncates or extends with empty slots
    for (let i = 0; i < size; i++) if (flat[i] === undefined) flat[i] = null;

    let idx = 0;
    const build = d => {
    const len = shape[d], arr = new Array(len);
    if (d === shape.length - 1) {
        for (let i = 0; i < len; i++) arr[i] = flat[idx++];
      } else {
        for (let i = 0; i < len; i++) arr[i] = build(d + 1);
      }
      return arr;
    };

    return build(0);
  }

  function fillTensor(tensor, val) {
    if (!Array.isArray(tensor)) return val;  
    return tensor.map(el => fillTensor(el, val));
  }

  function setTensorPath(tensor, path, value) {
    function f(a, d) {
      if (!Array.isArray(a)) return;
      const i = path[d] - 1;
      if (i < 0 || i >= a.length) return;
  
      const c = a.slice();
      if (d === path.length - 1) {
        c[i] = value;
      } else {
        const n = c[i];
        const r = f(n, d + 1);
        if (r === undefined) return;
        c[i] = r;
      }
      return c;
    }
  
    const out = f(tensor, 0);
    return out === undefined ? '' : new jwArray.Type(out);
  }

  function findTensorPath(t, target) {
  const stack = [[t, []]];

  while (stack.length) {
    const [node, path] = stack.pop();

    if (Array.isArray(node)) {
      for (let i = node.length; i--;) {
        stack.push([node[i], [...path, i]]);
      }
    } else if (node === target) {
      return path.map(i => i + 1);
    }
  }

  return [];
}

  function tensorContains(tensor, target) {
    if (Array.isArray(tensor)) {
      for (let i = 0; i < tensor.length; i++) {
        if (tensorContains(tensor[i], target)) return true;
      }
      return false;
    }
    return tensor === target;
  }

  function transposeTensor(t) {
    const shape = getTensorShape(t);
    if (!(Array.isArray(shape) && shape.length >= 1)) return [];
    const r = shape.length;
    if (r < 2) return t;
  
    const ns = shape.slice().reverse();
    const idx = new Array(r);
  
    function build(d) {
      const len = ns[d], out = new Array(len);
  
      if (d === r - 1) {
        for (let i = 0; i < len; i++) {
          idx[d] = i;
          let cur = t;
          for (let k = 0; k < r; k++) {
            cur = cur[idx[r - 1 - k]];
          }
          out[i] = cur;
        }
      } else {
        for (let i = 0; i < len; i++) {
          idx[d] = i;
          out[i] = build(d + 1);
        }
      }
  
      return out;
    }
  
    return build(0);
  }
  
  class Tensors {
    constructor() {
      if (!vm.jwArray) { vm.extensionManager.loadExtensionIdSync('jwArray'); }
      jwArray = vm.jwArray;
      runtime.registerCompiledExtensionBlocks('lxTensors', Tensors.compileInfo);
    }

    getInfo() {
      return {
        id: "lxTensors",
        name: "Tensors",
        color1: "#fe6743",
        menuIconURI: "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyMCAyMCI+PGNpcmNsZSBjeD0iMTAiIGN5PSIxMCIgcj0iOSIgc3R5bGU9InN0cm9rZS13aWR0aDoycHg7cGFpbnQtb3JkZXI6c3Ryb2tlO2ZpbGw6I2ZlNjc0MztzdHJva2U6I2NkM2IyYztmaWxsLXJ1bGU6bm9uemVybztmaWxsLW9wYWNpdHk6MSIvPjxwYXRoIGQ9Ik03LjcyNyA2Ljg0MmExLjMxIDEuMzEgMCAwIDAtMS4zMDUgMS4zMDR2My43MDhjMCAuNzE2LjU5IDEuMzA0IDEuMzA1IDEuMzA0SDkuMDN2LTEuNjgzaC0uOTI3di0yLjk1aC45MjdWNi44NDJabTMuMjQyIDB2MS42ODNoLjkyN3YyLjk1aC0uOTI3djEuNjgzaDEuMzA0YTEuMzEgMS4zMSAwIDAgMCAxLjMwNS0xLjMwNFY4LjE0NmExLjMxIDEuMzEgMCAwIDAtMS4zMDUtMS4zMDR6IiBzdHlsZT0iYmFzZWxpbmUtc2hpZnQ6YmFzZWxpbmU7ZGlzcGxheTppbmxpbmU7b3ZlcmZsb3c6dmlzaWJsZTtvcGFjaXR5OjE7dmVjdG9yLWVmZmVjdDpub25lO2ZpbGw6I2ZmZjtzdG9wLWNvbG9yOiMwMDA7c3RvcC1vcGFjaXR5OjEiLz48cGF0aCBmaWxsPSIjZmZmIiBkPSJNNy4yMzggNC4yMkg1LjMxMmExLjkyNyAxLjkyNyAwIDAgMC0xLjkyNyAxLjkyN3Y3LjcwNmMwIDEuMDY2Ljg2MyAxLjkyNyAxLjkyNyAxLjkyN2gxLjkyNnYtMS45MjdINS4zMTJWNi4xNDdoMS45MjZ6bTUuNTI0IDkuNjMzaDEuOTI2VjYuMTQ3aC0xLjkyNlY0LjIyaDEuOTI2YzEuMDY0IDAgMS45MjcuODYzIDEuOTI3IDEuOTI3djcuNzA2YTEuOTI2IDEuOTI2IDAgMCAxLTEuOTI3IDEuOTI3aC0xLjkyNnoiLz48L3N2Zz4=",
        blocks: [
          {
            opcode: 'blank',
            text: 'blank tensor',
            ...jwArray.Block
          },
          {
            opcode: 'blankSize',
            text: 'blank tensor of shape [SHA]',
            arguments: {
              SHA: {type: Scratch.ArgumentType.STRING, shape: Scratch.BlockShape.SQUARE, defaultValue: "[1, 2, 3]"},
            },
            ...jwArray.Block
          },
          '---',
          {
            opcode: 'tensorGetPath',
            text: 'get path [PAT] in tensor [TEN]',
            blockType: Scratch.BlockType.REPORTER,
            allowDropAnywhere: true,
            arguments: {
              PAT: {type: Scratch.ArgumentType.STRING, shape: Scratch.BlockShape.SQUARE, defaultValue: "[1, 2, 3]"},
              TEN: jwArray.Argument
            },
          },
          {
            opcode: 'tensorFindPath',
            text: 'path of [VAL] in tensor [TEN]',
            allowDropAnywhere: true,
            arguments: {
              VAL: {type: Scratch.ArgumentType.STRING, exemptFromNormalization: true, defaultValue: "foo"},
              TEN: jwArray.Argument
            },
            ...jwArray.Block
          },
          {
            opcode: 'tensorHas',
            text: 'tensor [TEN] has [VAL]',
            blockType: Scratch.BlockType.BOOLEAN,
            arguments: {
              TEN: jwArray.Argument,
              VAL: {type: Scratch.ArgumentType.STRING, exemptFromNormalization: true},
            },
          },
          {
            opcode: 'tensorShape',
            text: 'shape of tensor [TEN]',
            arguments: {
              TEN: jwArray.Argument
            },
            ...jwArray.Block
          },
          {
            opcode: 'tensorRank',
            text: 'rank of tensor [TEN]',
            blockType: Scratch.BlockType.REPORTER,
            arguments: {
              TEN: jwArray.Argument
            },
          },
          {
            opcode: 'tensorScalars',
            text: 'size of tensor [TEN]',
            blockType: Scratch.BlockType.REPORTER,
            arguments: {
              TEN: jwArray.Argument
            },
          },
          '---',
          {
            opcode: 'tensorSetPath',
            text: 'set path [PAT] in tensor [TEN] to [VAL]',
            arguments: {
              PAT: {type: Scratch.ArgumentType.STRING, shape: Scratch.BlockShape.SQUARE, defaultValue: "[1, 2, 3]"},
              TEN: jwArray.Argument,
              VAL: {type: Scratch.ArgumentType.STRING, exemptFromNormalization: true, defaultValue: "foo"}
            },
            ...jwArray.Block
          },
          {
            opcode: 'tensorReshape',
            text: 'reshape tensor [TEN] to shape [SHA]',
            arguments: {
              TEN: jwArray.Argument,
              SHA: {type: Scratch.ArgumentType.STRING, shape: Scratch.BlockShape.SQUARE, defaultValue: "[1, 2, 3]"},
            },
            ...jwArray.Block
          },
          {
            opcode: 'fill',
            text: 'fill tensor [TEN] with [VAL]',
            arguments: {
              TEN: jwArray.Argument,
              VAL: {type: Scratch.ArgumentType.STRING, exemptFromNormalization: true, defaultValue: "foo"}
            },
            ...jwArray.Block
          },
          {
            opcode: 'transpose',
            text: 'transpose tensor [TEN]',
            arguments: {
              TEN: jwArray.Argument
            },
            ...jwArray.Block
          },
          '---',
          {
            opcode: 'tensorValid',
            text: 'is [TEN] a valid tensor?',
            blockType: Scratch.BlockType.BOOLEAN,
            arguments: {
              TEN: {type: Scratch.ArgumentType.STRING, exemptFromNormalization: true}
            },
          },
        ],
      };
    }

    static compileInfo = {
      ir: {
        // blank(generator, block) {
        //   return {
        //     kind: 'input',
        //   };
        // },
        blankSize(generator, block) {
          return {
            kind: 'input',
            args: {
              SHA: generator.descendInputOfBlock(block, 'SHA'),
            }
          };
        },
        tensorGetPath(generator, block) {
          return {
            kind: 'input',
            args: {
              PAT: generator.descendInputOfBlock(block, 'PAT'),
              TEN: generator.descendInputOfBlock(block, 'TEN'),
            }
          };
        },
        tensorFindPath(generator, block) {
          return {
            kind: 'input',
            args: {
              VAL: generator.descendInputOfBlock(block, 'VAL'),
              TEN: generator.descendInputOfBlock(block, 'TEN'),
            }
          };
        },
        tensorHas(generator, block) {
          return {
            kind: 'input',
            args: {
              VAL: generator.descendInputOfBlock(block, 'VAL'),
              TEN: generator.descendInputOfBlock(block, 'TEN'),
            }
          };
        },
        tensorShape(generator, block) {
          return {
            kind: 'input',
            args: {
              TEN: generator.descendInputOfBlock(block, 'TEN'),
            }
          };
        },
      },
      js: {
        // blank(node, compiler, imports) {
        //   let source = '';
        //   source += `(`
        //   source += `new vm.jwArray.Type([], true)`
        //   source += `)`
        //   return new imports.TypedInput(source, imports.TYPE_UNKNOWN);
        // },
        blankSize(node, compiler, imports) {
          let source = '';

          source += compiler.script.yields ? `(yield* (function*(){` : `(function(){`;

          let dims = compiler.localVariables.next();
          let result = compiler.localVariables.next();
          source += `let ${dims} = vm.jwArray.Type.toArray(${compiler.descendInput(node.args.SHA).asUnknown()}, true).toJSON();`;
          source += `let ${result} = null;`;

          let i = compiler.localVariables.next();
          source += `for (let ${i} = ${dims}.length - 1; ${i} >= 0; ${i}--) {`
          source += `${result} = Array.from({ length: ${dims}[${i}] }, () => ${result} && structuredClone(${result}));`
          source += `}`

          source += `return vm.jwArray.Type.toArray(${result});`;

          source += compiler.script.yields ? `})())` : `})()`; // no semicolon

          return new imports.TypedInput(source, imports.TYPE_UNKNOWN);
        },
        tensorGetPath(node, compiler, imports) {
          let source = '';

          source += compiler.script.yields ? `(yield* (function*(){` : `(function(){`;

          let path = compiler.localVariables.next();
          let current = compiler.localVariables.next();
          
          source += `let ${current} = vm.jwArray.Type.toArray(${compiler.descendInput(node.args.TEN).asUnknown()}, true).toJSON();`
          source += `let ${path} = vm.jwArray.Type.toArray(${compiler.descendInput(node.args.PAT).asUnknown()}, true).toJSON();`
  
          let i = compiler.localVariables.next();
          let len = compiler.localVariables.next();
          source += `for (let ${i} = 0, ${len} = ${path}.length; ${i} < ${len}; ${i}++) {`
          source += `if (!Array.isArray(${current})) return '';`
          source += `${current} = ${current}[${path}[${i}]-1];`
          source += `if (${current} === undefined) return '';`
          source += `}`

          source += `return Array.isArray(${current}) ? vm.jwArray.Type.toArray(${current}) : ${current};`;

          source += compiler.script.yields ? `})())` : `})()`;

          return new imports.TypedInput(source, imports.TYPE_UNKNOWN);
        },
        tensorFindPath(node, compiler, imports) {
          let source = '';

          source += compiler.script.yields ? `(yield* (function*(){` : `(function(){`;

          let t = compiler.localVariables.next();
          let target = compiler.localVariables.next();
          
          source += `let ${t} = vm.jwArray.Type.toArray(${compiler.descendInput(node.args.TEN).asUnknown()}, true).toJSON();`
          source += `let ${target} = ${compiler.descendInput(node.args.VAL).asUnknown()};`

          let stack = compiler.localVariables.next();
          let nod = compiler.localVariables.next();
          let path = compiler.localVariables.next();

          source += `const ${stack} = [[${t}, []]];`

          source += `while (${stack}.length) {`
          source += `const [${nod}, ${path}] = ${stack}.pop();`

          let i = compiler.localVariables.next();
          source += `if (Array.isArray(${nod})) {`
          source += `for (let ${i} = ${nod}.length; ${i}--;) {`
          source += `${stack}.push([${nod}[${i}], [...${path}, ${i}]]);`
          source += `}`
          source += `} else if (${nod} === ${target}) {`
          source += `return vm.jwArray.Type.toArray(${path}.map(${i} => ${i} + 1));`
          source += `}`
          source += `}`

          source += `return new vm.jwArray.Type([], true);`

          source += compiler.script.yields ? `})())` : `})()`;

          return new imports.TypedInput(source, imports.TYPE_UNKNOWN);
        },
        tensorHas(node, compiler, imports) {
          let source = '';
          source += `(`
          source += `vm.jwArray.Type.toArray(${compiler.descendInput(node.args.TEN).asUnknown()}, true).flat(Infinity).toJSON().includes(${compiler.descendInput(node.args.VAL).asUnknown()})`
          source += `)`
          return new imports.TypedInput(source, imports.TYPE_UNKNOWN);
        },
        tensorShape(node, compiler, imports) {
          let source = '';
          source += compiler.script.yields ? `(yield* (function*(){` : `(function(){`;

          let ten = compiler.localVariables.next();
          source += `let ${a} = vm.jwArray.Type.toArray(${compiler.descendInput(node.args.TEN).asUnknown()}, true).toJSON();`

          let shape = compiler.localVariables.next();
          let length = compiler.localVariables.next();
          let first = compiler.localVariables.next();
          let type = compiler.localVariables.next();
          let idx = compiler.localVariables.next();
          let elem = compiler.localVariables.next();

          source += `const ${shape} = [];`
          source += `for (; Array.isArray(${ten}); ${ten} = ${ten}[0]) {`
          source += `  const ${length} = ${ten}.length;`
          source += `  if (!${length}) return [0];`
          source += `  ${shape}.push(${length});`
          source += `  const ${first} = ${ten}[0], ${type} = Array.isArray(${first});`
          source += `  for (let ${idx} = 1; ${idx} < ${length}; ${idx}++) {`
          source += `    const ${elem} = ${ten}[${idx}];`
          source += `    if (Array.isArray(${elem}) !== ${type} || (t && ${elem}.length !== ${first}.length)) return [];`
          source += `  }`
          source += `}`
          source += `return ${shape};`
          
          source += compiler.script.yields ? `})())` : `})()`;
          return new imports.TypedInput(source, imports.TYPE_UNKNOWN);
        },
      }
    }

    blank() {
      return new jwArray.Type(getTensorShape(u(TEN)));
    }
    tensorShape({ TEN }) {
      TEN = jwArray.Type.toArray(TEN);
      if (TEN.array == null) return new jwArray.Type([], true);
      return new jwArray.Type(getTensorShape(u(TEN)));
    }
    tensorRank({ TEN }) {
      TEN = jwArray.Type.toArray(TEN);
      if (TEN.array == null) return '';
      return getTensorShape(u(TEN)).length;
    }
    tensorScalars({ TEN }) {
      TEN = jwArray.Type.toArray(TEN);
      if (TEN.array == null) return '';
      return getTensorShape(u(TEN)).reduce((a, b) => a*b, 1);
    }

    tensorSetPath({ PAT, TEN, VAL }) {
      TEN = jwArray.Type.toArray(TEN);
      PAT = jwArray.Type.toArray(PAT);
      if (TEN.array == null || (Array.isArray(TEN.array) && TEN.array.length === 0)) return new jwArray.Type([], true);
      return setTensorPath(u(TEN), PAT.array, VAL)
    }
    tensorReshape({ TEN, SHA }) {
      TEN = jwArray.Type.toArray(TEN);
      SHA = jwArray.Type.toArray(SHA);
      if (TEN.array == null || (Array.isArray(TEN.array) && TEN.array.length === 0)) return new jwArray.Type([], true);
      return new jwArray.Type(reshapeTensor(u(TEN), SHA.array));
    }
    fill({ TEN, VAL }) {
      TEN = jwArray.Type.toArray(TEN);
      if (TEN.array == null || (Array.isArray(TEN.array) && TEN.array.length === 0)) return new jwArray.Type([], true);
      return new jwArray.Type(fillTensor(u(TEN), VAL));
    }
    transpose({ TEN }) {
      TEN = jwArray.Type.toArray(TEN);
      if (TEN.array == null) return new jwArray.Type([], true);
      return new jwArray.Type(transposeTensor(u(TEN)));
    }

    tensorValid({ TEN }) {
      if (TEN == "" || TEN == null) return false;
      TEN = jwArray.Type.toArray(TEN);
      const arr = TEN?.array;
      if (!Array.isArray(arr)) return false;
      TEN = getTensorShape(u(TEN))
      return (Array.isArray(TEN) && TEN.length >= 1);
    }
  }
  
  Scratch.extensions.register(new Tensors());
})(Scratch);
