(function(Scratch) {
  'use strict';

  let arrayLimit = 2 ** 32 - 1;

  function formatNumber(x) {
      if (x >= 1e6) {
          return x.toExponential(4);
      } else {
          x = Math.floor(x * 1000) / 1000;
          return x.toFixed(Math.min(3, (String(x).split('.')[1] || '').length));
      }
  }

  const escapeHTML = unsafe => {
      return unsafe
          .replaceAll("&", "&amp;")
          .replaceAll("<", "&lt;")
          .replaceAll(">", "&gt;")
          .replaceAll('"', "&quot;")
          .replaceAll("'", "&#039;");
  };

  function clampIndex(x) {
      return Math.min(Math.max(Math.floor(x), 0), arrayLimit);
  }

  function span(text) {
      let el = document.createElement('span')
      el.innerHTML = text
      el.style.display = 'hidden'
      el.style.whiteSpace = 'nowrap'
      el.style.width = '100%'
      el.style.textAlign = 'center'
      return el
  }

  function isObject(x) {
      return x !== null && typeof x === "object" && [null, Object.prototype].includes(Object.getPrototypeOf(x));
  }

  const vm = Scratch.vm
  const runtime = vm.runtime

  const u = x =>
    x instanceof jwArray.Type ? u(x.toJSON()) :
    Array.isArray(x) ? x.map(u) :
    x;

  let TensorType, lxTensor, jwArray;

  // TODO: put these all in tensor class

  // function findTensorPath(t, target) {
  // const stack = [[t, []]];

  // while (stack.length) {
  //   const [node, path] = stack.pop();

  //   if (Array.isArray(node)) {
  //     for (let i = node.length; i--;) {
  //       stack.push([node[i], [...path, i]]);
  //     }
  //   } else if (node === target) {
  //     return path.map(i => i + 1);
  //   }
  // }

  // return [];
  // }

  // function tensorContains(tensor, target) {
  //   if (Array.isArray(tensor)) {
  //     for (let i = 0; i < tensor.length; i++) {
  //       if (tensorContains(tensor[i], target)) return true;
  //     }
  //     return false;
  //   }
  //   return tensor === target;
  // }

  // function transposeTensor(t) {
  //   const shape = getTensorShape(t);
  //   if (!(Array.isArray(shape) && shape.length >= 1)) return [];
  //   const r = shape.length;
  //   if (r < 2) return t;

  //   const ns = shape.slice().reverse();
  //   const idx = new Array(r);

  //   function build(d) {
  //     const len = ns[d], out = new Array(len);

  //     if (d === r - 1) {
  //       for (let i = 0; i < len; i++) {
  //         idx[d] = i;
  //         let cur = t;
  //         for (let k = 0; k < r; k++) {
  //           cur = cur[idx[r - 1 - k]];
  //         }
  //         out[i] = cur;
  //       }
  //     } else {
  //       for (let i = 0; i < len; i++) {
  //         idx[d] = i;
  //         out[i] = build(d + 1);
  //       }
  //     }

  //     return out;
  //   }

  //   return build(0);
  // }
  
  class Tensors {
    constructor() {
      if (!vm.jwArray) { vm.extensionManager.loadExtensionIdSync('jwArray'); }
      jwArray = vm.jwArray;

      TensorType = class extends jwArray.Type {
        customId = "lxTensor"

        constructor(array = [], safe = false, shape) {
          super(array, safe);
          this.array = safe ? array : array.map(v => {
            if (v instanceof Array) return new TensorType([...v])
            if (vm.dogeiscutObject && isObject(v)) return new vm.dogeiscutObject.Type({...v})
            return v
          })
          this._shape = shape ?? TensorType.shape(array);
        }

        static toTensor(x, readOnly = false) {
          if (x instanceof jwArray.Type) return readOnly ? x : new TensorType([...x.array], true)
          if (x instanceof Array) return readOnly ? new TensorType(x) : new TensorType([...x])
          if (x === "" || x === null || x === undefined) return new TensorType([], true)
          if (typeof x == "object" && typeof x.toJSON == "function") {
              let parsed = x.toJSON()
              if (parsed instanceof Array) return new TensorType(parsed)
              if (isObject(parsed)) return new TensorType(Object.values(parsed))
              return new TensorType([parsed])
          }
          try {
              let parsed = JSON.parse(x)
              if (parsed instanceof Array) return new TensorType(parsed)
          } catch {}
          return new TensorType([x], true)
        }

        jwArrayHandler() {
          return `Tensor<${formatNumber(this.array.length)}>`
        }

        toReporterContent() {
          let root = document.createElement('div')
          root.style.display = 'flex'
          root.style.flexDirection = 'column'
          root.style.justifyContent = 'center'

          let arrayDisplay = span(`[${this.array.slice(0, 50).map(v => TensorType.display(v)).join(', ')}]`)
          arrayDisplay.style.overflow = "hidden"
          arrayDisplay.style.whiteSpace = "nowrap"
          arrayDisplay.style.textOverflow = "ellipsis"
          arrayDisplay.style.maxWidth = "256px"
          root.appendChild(arrayDisplay)

          root.appendChild(span(`Length: ${this.array.length}`))
          let shape = Array.isArray(this.shape) ? (this.shape.length === 0 ? '?' : this.shape) : [];
          shape = Array.isArray(shape) ? `[${shape.slice(0, 50).map(v => TensorType.display(v)).join(', ')}]` : '?';
          root.appendChild(span(`Shape: ${shape}`));

          return root
        }

        get shape() {
          if (!Array.isArray(this._shape)) {
            this._shape = TensorType.shape(this);
          }
          return this._shape;
        }

        static shape(a) {
          const s = [];
          a = u(a);
          for (; Array.isArray(a); a = u(a[0])) {
            const l = a.length;
            if (!l) return [0];
            s.push(l);
            const f = u(a[0]);
            const t = Array.isArray(f);
            for (let i = 1; i < l; i++) {
              const x = u(a[i]);
              if (Array.isArray(x) !== t || (t && x.length !== f.length)) return [];
            }
          }
          return s;
        }

        static blankSize(shape) {
          shape = u(shape);
          let result = null;

          for (let i = shape.length - 1; i >= 0; i--) {
            result = Array.from({ length: shape[i] }, () => result && structuredClone(result));
          }

          return new TensorType(result, false, shape);
        }

        static isEmpty(t) {
          return t instanceof jwArray.Type && t.array.length === 0;
        }

        fillTensor(val) {
          if (TensorType.isEmpty(this)) return this;
          const fill = (x) => {
            if (Array.isArray(x)) {
              for (let i = 0; i < x.length; i++) {
                const el = x[i];

                if (el instanceof TensorType) {
                  fill(el.array); //inside tenosr clas 
                } else if (Array.isArray(el)) {
                  fill(el);
                } else {
                  x[i] = val;
                }
              }
            }
          };

          fill(this.array);
          return this;
        }

        reshape(shape) {
          if (TensorType.isEmpty(this)) return this;
          shape = u(shape);
          if (!Array.isArray(shape) || shape.some(n => n <= 0 || n !== n)) {
            this.array = [];
            this._shape = [];
            return this;
          }

          const size = shape.reduce((a, b) => a * b, 1);

          const flat = [];
          const f = (x) => {
            x = u(x);
            if (Array.isArray(x)) {
              for (let i = 0; i < x.length && flat.length < size; i++) {
                f(x[i]);
              }
            } else if (flat.length < size) { // trunc
              flat.push(x);
            }
          };
          f(this);

          while (flat.length < size) flat.push(null); // extend

          let idx = 0;

          const build = (d) => {
            const len = shape[d];

            // leaf
            if (d === shape.length - 1) {
              const arr = new Array(len);
              for (let i = 0; i < len; i++) arr[i] = flat[idx++];
              return arr;
            }

            const subShape = shape.slice(d + 1);
            const arr = new Array(len);

            for (let i = 0; i < len; i++) {
              arr[i] = new TensorType(build(d + 1), true, subShape);
            }

            return arr;
          };

          this.array = build(0);
          this._shape = shape;

          return this;
        }

        setPath(path, value) {
          if (TensorType.isEmpty(this)) return this;
          path = u(path);
          let node = this.array;
          
          for (let d = 0; d < path.length - 1; d++) {
              if (node instanceof TensorType) node = node.array;
              
              if (!Array.isArray(node)) return this;
              
              const i = Number(path[d]) - 1;
              if (i < 0 || i >= node.length) return this;
              
              node = node[i];
          }
          
          if (node instanceof TensorType) node = node.array;
          if (!Array.isArray(node)) return this;
          
          const i = Number(path[path.length - 1]) - 1;
          if (i < 0 || i >= node.length) return this;
          
          node[i] = value;
          return this;
        }

        getPath(path) {
          if (TensorType.isEmpty(this)) return undefined;
          path = u(path);
          let node = this.array;

          for (let d = 0; d < path.length; d++) {
            if (node instanceof TensorType) node = node.array;

            if (!Array.isArray(node)) return undefined;

            const i = Number(path[d]) - 1;
            if (i < 0 || i >= node.length) return undefined;

            node = node[i];
          }

          if (node instanceof TensorType) return node.array;

          return node;
        }

        findPath(target) {
          if (TensorType.isEmpty(this)) return this;
          target = u(target);

          const eq = (a, b) => {
            a = u(a); b = u(b);
            if (Array.isArray(a) !== Array.isArray(b)) return false;
            if (!Array.isArray(a)) return a === b;
            if (a.length !== b.length) return false;
            for (let i = 0; i < a.length; i++) {
              if (!eq(a[i], b[i])) return false;
            }
            return true;
          };

          const stack = [[this.array, []]];

          while (stack.length) {
            let [node, path] = stack.pop();
            if (node instanceof TensorType) node = node.array;

            node = u(node);

            if (eq(node, target)) return new jwArray.Type(path.map(i => i + 1), true);

            if (Array.isArray(node)) {
              for (let i = node.length; i--; ) {
                stack.push([node[i], [...path, i]]);
              }
            }
          }

          return new jwArray.Type([], true);
        }

        flatHas(val) {
          if (TensorType.isEmpty(this)) return this;
          val = u(val);
          return this.flat(Infinity).some(el => u(el) === val);
        }
      }

      lxTensor = {
        Type: TensorType,
        Block: {
          blockType: Scratch.BlockType.REPORTER,
          blockShape: Scratch.BlockShape.SQUARE,
          forceOutputType: ["Array", "Tensor"],
          disableMonitor: true
        },
        Argument: {
          shape: Scratch.BlockShape.SQUARE,
          exemptFromNormalization: true,
          check: ["Array"]
        }
      };

      vm.lxTensor = lxTensor;
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
            ...lxTensor.Block
          },
          {
            opcode: 'blankSize',
            text: 'blank tensor of shape [SHA]',
            arguments: {
              SHA: {type: Scratch.ArgumentType.STRING, shape: Scratch.BlockShape.SQUARE, defaultValue: "[1, 2, 3]"},
            },
            ...lxTensor.Block
          },
          {
            opcode: 'parse',
            text: 'parse [STR] as tensor',
            arguments: {
              STR: {type: Scratch.ArgumentType.STRING, defaultValue: "[[1, 2], [3, 4]]"},
            },
            ...lxTensor.Block
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
            text: '(wip) tensor [TEN] has [VAL]',
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
            text: '(wip) transpose tensor [TEN]',
            arguments: {
              TEN: jwArray.Argument
            },
            ...jwArray.Block
          },
          '---',
          {
            opcode: 'tensorValid',
            text: '(is [TEN] a valid tensor?',
            blockType: Scratch.BlockType.BOOLEAN,
            arguments: {
              TEN: {type: Scratch.ArgumentType.STRING, exemptFromNormalization: true}
            },
          },
        ],
      };
    }

    // TODO: use a custom type blud

    static compileInfo = {
      ir: {
        blank(generator, block) {
          return {
            kind: 'input',
          };
        },
    //     blankSize(generator, block) {
    //       return {
    //         kind: 'input',
    //         args: {
    //           SHA: generator.descendInputOfBlock(block, 'SHA'),
    //         }
    //       };
    //     },
    //     tensorGetPath(generator, block) {
    //       return {
    //         kind: 'input',
    //         args: {
    //           PAT: generator.descendInputOfBlock(block, 'PAT'),
    //           TEN: generator.descendInputOfBlock(block, 'TEN'),
    //         }
    //       };
    //     },
    //     tensorFindPath(generator, block) {
    //       return {
    //         kind: 'input',
    //         args: {
    //           VAL: generator.descendInputOfBlock(block, 'VAL'),
    //           TEN: generator.descendInputOfBlock(block, 'TEN'),
    //         }
    //       };
    //     },
    //     tensorHas(generator, block) {
    //       return {
    //         kind: 'input',
    //         args: {
    //           VAL: generator.descendInputOfBlock(block, 'VAL'),
    //           TEN: generator.descendInputOfBlock(block, 'TEN'),
    //         }
    //       };
    //     },
    //     tensorShape(generator, block) {
    //       return {
    //         kind: 'input',
    //         args: {
    //           TEN: generator.descendInputOfBlock(block, 'TEN'),
    //         }
    //       };
    //     },
      },
      js: {
        blank(node, compiler, imports) {
          let source = '';
          source += `(new vm.lxTensor.Type([], true))`;
          return new imports.TypedInput(source, imports.TYPE_UNKNOWN);
        },
    //     blankSize(node, compiler, imports) {
    //       let source = '';

    //       source += compiler.script.yields ? `(yield* (function*(){` : `(function(){`;

    //       let dims = compiler.localVariables.next();
    //       let result = compiler.localVariables.next();
    //       source += `let ${dims} = vm.jwArray.Type.toArray(${compiler.descendInput(node.args.SHA).asUnknown()}, true).toJSON();`;
    //       source += `let ${result} = null;`;

    //       let i = compiler.localVariables.next();
    //       source += `for (let ${i} = ${dims}.length - 1; ${i} >= 0; ${i}--) {`
    //       source += `${result} = Array.from({ length: ${dims}[${i}] }, () => ${result} && structuredClone(${result}));`
    //       source += `}`

    //       source += `return vm.jwArray.Type.toArray(${result});`;

    //       source += compiler.script.yields ? `})())` : `})()`; // no semicolon

    //       return new imports.TypedInput(source, imports.TYPE_UNKNOWN);
    //     },
    //     tensorGetPath(node, compiler, imports) {
    //       let source = '';

    //       source += compiler.script.yields ? `(yield* (function*(){` : `(function(){`;

    //       let path = compiler.localVariables.next();
    //       let current = compiler.localVariables.next();
          
    //       source += `let ${current} = vm.jwArray.Type.toArray(${compiler.descendInput(node.args.TEN).asUnknown()}, true).toJSON();`
    //       source += `let ${path} = vm.jwArray.Type.toArray(${compiler.descendInput(node.args.PAT).asUnknown()}, true).toJSON();`
  
    //       let i = compiler.localVariables.next();
    //       let len = compiler.localVariables.next();
    //       source += `for (let ${i} = 0, ${len} = ${path}.length; ${i} < ${len}; ${i}++) {`
    //       source += `if (!Array.isArray(${current})) return '';`
    //       source += `${current} = ${current}[${path}[${i}]-1];`
    //       source += `if (${current} === undefined) return '';`
    //       source += `}`

    //       source += `return Array.isArray(${current}) ? vm.jwArray.Type.toArray(${current}) : ${current};`;

    //       source += compiler.script.yields ? `})())` : `})()`;

    //       return new imports.TypedInput(source, imports.TYPE_UNKNOWN);
    //     },
    //     tensorFindPath(node, compiler, imports) {
    //       let source = '';

    //       source += compiler.script.yields ? `(yield* (function*(){` : `(function(){`;

    //       let t = compiler.localVariables.next();
    //       let target = compiler.localVariables.next();
          
    //       source += `let ${t} = vm.jwArray.Type.toArray(${compiler.descendInput(node.args.TEN).asUnknown()}, true).toJSON();`
    //       source += `let ${target} = ${compiler.descendInput(node.args.VAL).asUnknown()};`

    //       let stack = compiler.localVariables.next();
    //       let nod = compiler.localVariables.next();
    //       let path = compiler.localVariables.next();

    //       source += `const ${stack} = [[${t}, []]];`

    //       source += `while (${stack}.length) {`
    //       source += `const [${nod}, ${path}] = ${stack}.pop();`

    //       let i = compiler.localVariables.next();
    //       source += `if (Array.isArray(${nod})) {`
    //       source += `for (let ${i} = ${nod}.length; ${i}--;) {`
    //       source += `${stack}.push([${nod}[${i}], [...${path}, ${i}]]);`
    //       source += `}`
    //       source += `} else if (${nod} === ${target}) {`
    //       source += `return vm.jwArray.Type.toArray(${path}.map(${i} => ${i} + 1));`
    //       source += `}`
    //       source += `}`

    //       source += `return new vm.jwArray.Type([], true);`

    //       source += compiler.script.yields ? `})())` : `})()`;

    //       return new imports.TypedInput(source, imports.TYPE_UNKNOWN);
    //     },
    //     tensorHas(node, compiler, imports) {
    //       let source = '';
    //       source += `(`
    //       source += `vm.jwArray.Type.toArray(${compiler.descendInput(node.args.TEN).asUnknown()}, true).flat(Infinity).toJSON().includes(${compiler.descendInput(node.args.VAL).asUnknown()})`
    //       source += `)`
    //       return new imports.TypedInput(source, imports.TYPE_UNKNOWN);
    //     },
    //     tensorShape(node, compiler, imports) {
    //       let source = '';
    //       source += compiler.script.yields ? `(yield* (function*(){` : `(function(){`;

    //       let ten = compiler.localVariables.next();
    //       source += `let ${a} = vm.jwArray.Type.toArray(${compiler.descendInput(node.args.TEN).asUnknown()}, true).toJSON();`

    //       let shape = compiler.localVariables.next();
    //       let length = compiler.localVariables.next();
    //       let first = compiler.localVariables.next();
    //       let type = compiler.localVariables.next();
    //       let idx = compiler.localVariables.next();
    //       let elem = compiler.localVariables.next();

    //       source += `const ${shape} = [];`
    //       source += `for (; Array.isArray(${ten}); ${ten} = ${ten}[0]) {`
    //       source += `  const ${length} = ${ten}.length;`
    //       source += `  if (!${length}) return [0];`
    //       source += `  ${shape}.push(${length});`
    //       source += `  const ${first} = ${ten}[0], ${type} = Array.isArray(${first});`
    //       source += `  for (let ${idx} = 1; ${idx} < ${length}; ${idx}++) {`
    //       source += `    const ${elem} = ${ten}[${idx}];`
    //       source += `    if (Array.isArray(${elem}) !== ${type} || (t && ${elem}.length !== ${first}.length)) return [];`
    //       source += `  }`
    //       source += `}`
    //       source += `return ${shape};`
          
    //       source += compiler.script.yields ? `})())` : `})()`;
    //       return new imports.TypedInput(source, imports.TYPE_UNKNOWN);
        // },
      }
    }

    blank() {
      return new lxTensor.Type([], true);
    }
    blankSize({ SHA }) {
      SHA = jwArray.Type.toArray(SHA);
      return lxTensor.Type.blankSize(SHA);
    }
    parse({ STR }) {
      return lxTensor.Type.toTensor(STR);
    }

    tensorGetPath({ PAT, TEN }) {
      TEN = lxTensor.Type.toTensor(TEN);
      PAT = jwArray.Type.toArray(PAT);
      const res = TEN.getPath(PAT);
      return (res === undefined) ? '' : res;
    }
    tensorFindPath({ VAL, TEN }) {
      TEN = lxTensor.Type.toTensor(TEN);
      return TEN.findPath(VAL);
    }
    tensorShape({ TEN }) {
      TEN = lxTensor.Type.toTensor(TEN);
      return new jwArray.Type(TEN.shape);
    }
    tensorRank({ TEN }) {
      TEN = lxTensor.Type.toTensor(TEN);
      if (TEN.array == null) return 0;
      return TEN.shape.length;
    }
    tensorScalars({ TEN }) {
      TEN = lxTensor.Type.toTensor(TEN);
      if (TEN.array == null) return 0;
      return TEN.flat(Infinity).array.length;
    }

    tensorSetPath({ PAT, TEN, VAL }) {
      TEN = lxTensor.Type.toTensor(TEN);
      PAT = jwArray.Type.toArray(PAT);
      return TEN.setPath(PAT, VAL);
    }
    tensorReshape({ TEN, SHA }) {
      TEN = lxTensor.Type.toTensor(TEN);
      SHA = jwArray.Type.toArray(SHA);
      return TEN.reshape(SHA.array);
    }
    fill({ TEN, VAL }) {
      TEN = lxTensor.Type.toTensor(TEN);
      return TEN.fillTensor(VAL)
    }
    transpose({ TEN }) {
      TEN = lxTensor.Type.toTensor(TEN);
      return new jwArray.Type(transposeTensor(u(TEN)));
    }

    tensorValid({ TEN }) {
      if (TEN == "" || TEN == null) return false;
      TEN = lxTensor.Type.toTensor(TEN);
      return TEN.shape.length > 0 && TEN.array.length > 0;
    }
  }
  
  Scratch.extensions.register(new Tensors());
})(Scratch);