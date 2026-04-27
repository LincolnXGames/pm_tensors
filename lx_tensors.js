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

  function isPlainObject(value) {
    if (typeof value !== 'object' || value === null) {
      return false
    }

    const prototype = Object.getPrototypeOf(value)
    return (prototype === null || prototype === Object.prototype || Object.getPrototypeOf(prototype) === null) && !(Symbol.toStringTag in value) && !(Symbol.iterator in value)
  }

  const vm = Scratch.vm
  const runtime = vm.runtime

  const u = x =>
    x instanceof jwArray.Type ? u(x.toJSON()) :
    Array.isArray(x) ? x.map(u) :
    x;

  let TensorType, lxTensor, jwArray;
  
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

        static parseLength(x) {
          if (u(x) instanceof Array) return u(x).map(TensorType.parseLength);
          return Math.min(Math.max(Math.floor(x), 0), arrayLimit) || 0;
        }

        jwArrayHandler() {
          return `Tensor<${formatNumber(this.array.length)}>`
        }

        toReporterContent() {
          return TensorType.tableDisplay(this);
        }

        toMonitorContent() {
          return TensorType.tableDisplay(this, '1px solid #fff', '#ffffff33', 'ffffff00');
        }

        static tableDisplay(source, border = '1px solid #77777777', keyBackground = '#77777724', background = '#ffffff00', entryLimit = 1000) {
          const ogSource = source;

          let root = document.createElement('div')
          root.style.display = 'flex'
          root.style.flexDirection = 'column'

          const renderArray = (array, type = 'Array') => {
            const table = document.createElement('table')
            table.style.borderCollapse = 'collapse'
            table.style.margin = '2px 0'
            table.style.fontSize = '12px'
            table.style.background = background
            table.style.border = border

            const limitedArray = array.slice(0, entryLimit)

            if (limitedArray.length === 0) {
              const text = span(`<i style="opacity: 0.75;">${escapeHTML(`<Blank ${type}>`)}</i>`)

              return text.outerHTML
            }

            limitedArray.forEach((value, index) => {
              const centeringDiv = document.createElement('div')
              centeringDiv.style.display = 'flex'
              centeringDiv.style.justifyContent = 'center'

              const row = document.createElement('tr')

              const valueCell = document.createElement('td')
              valueCell.style.border = border
              valueCell.style.padding = '2px 6px'
              valueCell.style.background = background

              centeringDiv.innerHTML = render(value, border, keyBackground, background, entryLimit)

              valueCell.appendChild(centeringDiv)
              row.appendChild(valueCell)
              table.appendChild(row)
            })

            if (array.length > entryLimit) {
              const moreRow = document.createElement('tr')
              const moreCell = document.createElement('td')
              moreCell.colSpan = 2
              moreCell.textContent = `... ${array.length - entryLimit} more values`
              moreCell.style.textAlign = 'center'
              moreCell.style.fontStyle = 'italic'
              moreCell.style.color = border
              moreRow.appendChild(moreCell)
              table.appendChild(moreRow)
            }

            return table.outerHTML
          }

          const renderMap = (map) => {
            const table = document.createElement('table')
            table.style.borderCollapse = 'collapse'
            table.style.margin = '2px 0'
            table.style.fontSize = '12px'
            table.style.background = background
            table.style.border = border

            const limitedMap = new Map(Array.from(map).slice(0, entryLimit))

            if (limitedMap.size === 0) {
              const text = span(`<i style="opacity: 0.75;">${escapeHTML("<Blank Object>")}</i>`)

              return text.outerHTML
            }

            limitedMap.forEach((value, key) => {
              const keyCenteringDiv = document.createElement('div')
              keyCenteringDiv.style.display = 'flex'
              keyCenteringDiv.style.justifyContent = 'center'

              const valueCenteringDiv = document.createElement('div')
              valueCenteringDiv.style.display = 'flex'
              valueCenteringDiv.style.justifyContent = 'center'

              const row = document.createElement('tr')

              const keyCell = document.createElement('td')
              keyCell.style.border = border
              keyCell.style.padding = '2px 6px'
              keyCell.style.background = keyBackground
              keyCell.style.fontWeight = 'bold';

              keyCenteringDiv.innerHTML = renderKey(key)

              const valueCell = document.createElement('td')
              valueCell.style.border = border
              valueCell.style.padding = '2px 6px'
              valueCell.style.background = background

              valueCenteringDiv.innerHTML = render(value, border, keyBackground, background)

              keyCell.appendChild(keyCenteringDiv)
              row.appendChild(keyCell)
              valueCell.appendChild(valueCenteringDiv)
              row.appendChild(valueCell)
              table.appendChild(row)
            })

            if (map.size > entryLimit) {
              const moreRow = document.createElement('tr')
              const moreCell = document.createElement('td')
              moreCell.colSpan = 2
              moreCell.textContent = `... ${map.size - entryLimit} more entries`
              moreCell.style.textAlign = 'center'
              moreCell.style.fontStyle = 'italic'
              moreCell.style.color = border
              moreRow.appendChild(moreCell)
              table.appendChild(moreRow)
            }
            
            return table.outerHTML
          }

          const renderKey = (x) => {
            if (typeof x === "symbol") {
              return `<i style="opacity: 0.5;">${escapeHTML(x.description)}</i>`
            }
            return escapeHTML(String(x))
          }

          const render = (x) => {
            try {
              const nullDraw = '<i style="opacity: 0.75;">null</i>'
              switch (typeof x) {
                case "object":
                  if (x === null || x === undefined) return nullDraw
                  if (x instanceof Array) {
                      return renderArray(x, ogSource instanceof TensorType ? 'Tensor' : 'Array')
                  }
                  if (x instanceof Map) {
                      return renderMap(x)
                  }
                  if (typeof x.dogeiscutObjectHandler == "function") {
                      return x.dogeiscutObjectHandler(x)
                  }
                  if (typeof x.jwArrayHandler == "function") {
                      return x.jwArrayHandler(x)
                  }
                  return "Object"
                case "undefined":
                  return nullDraw
                case "number":
                  return formatNumber(x)
                case "boolean":
                  return x ? "true" : "false"
                case "string":
                  return `"${escapeHTML(Scratch.Cast.toString(x))}"`
                case "symbol":
                  return `<i style="opacity: 0.5;">${escapeHTML(x.description)}</i>`
              }
            } catch {}
            return "?"
          }

          const normalize = (input) => {
            if (input instanceof jwArray.Type) {
              return input.array.map(v => normalize(v))
            }
            if (vm.dogeiscutObject && input instanceof vm.dogeiscutObject.Type) {
              return new Map(Array.from(input.map).map(([k, v]) => [vm.dogeiscutObject.Type.forKey(k), normalize(v)]))
            }
            if (isPlainObject(input)) {
              return new Map(Object.entries(input).map(([k, v]) => [vm.dogeiscutObject.Type.forKey(k), normalize(v)]))
            }
            return input
          }

          source = normalize(source)

          root.innerHTML = render(source, border, keyBackground, background)
          root.appendChild(span(`Length: ${source.length}`))
          root.appendChild(span(`Shape: [${ogSource.shape.join(', ')}]`))

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
          shape = TensorType.parseLength(shape);
          let result = null;

          for (let i = shape.length - 1; i >= 0; i--) {
            result = Array.from({ length: shape[i] }, () => result && structuredClone(result));
          }

          return new TensorType(result, false, shape);
        }

        static isEmpty(t) {
          return !t instanceof jwArray.Type || t?.array?.length === 0;
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
          if (TensorType.isEmpty(this) || this.shape == shape) return this;
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
          if (TensorType.isEmpty(this)) return false;
          val = u(val);
          return u(this.flat(Infinity)).includes(val);
        }

        transpose() {
          if (TensorType.isEmpty(this)) return this;

          const t = u(this);
          const shape = this.shape;

          if (!Array.isArray(shape) || shape.length < 2) return this;

          const r = shape.length;
          const newShape = shape.slice().reverse();
          const idx = new Array(r);

          const build = (d) => {
            const len = newShape[d];
            const out = new Array(len);

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
          };

          const result = build(0);

          this.array = result;
          this._shape = newShape;

          return this;
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
            opcode: 'getPath',
            text: 'get path [PAT] in tensor [TEN]',
            blockType: Scratch.BlockType.REPORTER,
            allowDropAnywhere: true,
            arguments: {
              PAT: {type: Scratch.ArgumentType.STRING, shape: Scratch.BlockShape.SQUARE, defaultValue: "[1, 2, 3]"},
              TEN: jwArray.Argument
            },
          },
          {
            opcode: 'findPath',
            text: 'path of [VAL] in tensor [TEN]',
            allowDropAnywhere: true,
            arguments: {
              VAL: {type: Scratch.ArgumentType.STRING, exemptFromNormalization: true, defaultValue: "foo"},
              TEN: jwArray.Argument
            },
            ...lxTensor.Block
          },
          {
            opcode: 'has',
            text: 'tensor [TEN] has [VAL]',
            blockType: Scratch.BlockType.BOOLEAN,
            arguments: {
              TEN: jwArray.Argument,
              VAL: {type: Scratch.ArgumentType.STRING, exemptFromNormalization: true, defaultValue: "foo"},
            },
          },
          {
            opcode: 'shape',
            text: 'shape of tensor [TEN]',
            arguments: {
              TEN: jwArray.Argument
            },
            ...lxTensor.Block
          },
          {
            opcode: 'rank',
            text: 'rank of tensor [TEN]',
            blockType: Scratch.BlockType.REPORTER,
            arguments: {
              TEN: jwArray.Argument
            },
          },
          {
            opcode: 'scalars',
            text: 'size of tensor [TEN]',
            blockType: Scratch.BlockType.REPORTER,
            arguments: {
              TEN: jwArray.Argument
            },
          },
          '---',
          {
            opcode: 'setPath',
            text: 'set path [PAT] in tensor [TEN] to [VAL]',
            arguments: {
              PAT: {type: Scratch.ArgumentType.STRING, shape: Scratch.BlockShape.SQUARE, defaultValue: "[1, 2, 3]"},
              TEN: jwArray.Argument,
              VAL: {type: Scratch.ArgumentType.STRING, exemptFromNormalization: true, defaultValue: "foo"}
            },
            ...lxTensor.Block
          },
          {
            opcode: 'reshape',
            text: 'reshape tensor [TEN] to shape [SHA]',
            arguments: {
              TEN: jwArray.Argument,
              SHA: {type: Scratch.ArgumentType.STRING, shape: Scratch.BlockShape.SQUARE, defaultValue: "[1, 2, 3]"},
            },
            ...lxTensor.Block
          },
          {
            opcode: 'fill',
            text: 'fill tensor [TEN] with [VAL]',
            arguments: {
              TEN: jwArray.Argument,
              VAL: {type: Scratch.ArgumentType.STRING, exemptFromNormalization: true, defaultValue: "foo"}
            },
            ...lxTensor.Block
          },
          {
            opcode: 'transpose',
            text: 'transpose tensor [TEN]',
            arguments: {
              TEN: jwArray.Argument
            },
            ...lxTensor.Block
          },
          '---',
          {
            opcode: 'mapP',
            text: 'path',
            blockType: Scratch.BlockType.REPORTER,
            hideFromPalette: true,
            canDragDuplicate: true
          },
          {
            opcode: 'mapV',
            text: 'value',
            blockType: Scratch.BlockType.REPORTER,
            hideFromPalette: true,
            allowDropAnywhere: true,
            canDragDuplicate: true
          },
          {
            opcode: 'mapV2',
            text: 'value2',
            blockType: Scratch.BlockType.REPORTER,
            hideFromPalette: true,
            allowDropAnywhere: true,
            canDragDuplicate: true
          },
          {
            opcode: 'map',
            text: 'map [TEN] [P] [V] = [VAL]',
            arguments: {
              TEN: jwArray.Argument,
              P: {fillIn: 'mapP'},
              V: {fillIn: 'mapV'},
              VAL: {type: Scratch.ArgumentType.STRING, exemptFromNormalization: true, defaultValue: "foo"}
            },
            ...lxTensor.Block
          },
          {
            opcode: 'combine',
            text: 'combine [TEN] with [TEN2] [P] [V] [V2] = [VAL]',
            arguments: {
              TEN: jwArray.Argument,
              TEN2: jwArray.Argument,
              P: {fillIn: 'mapP'},
              V: {fillIn: 'mapV'},
              V2: {fillIn: 'mapV2'},
              VAL: {type: Scratch.ArgumentType.STRING, exemptFromNormalization: true, defaultValue: "foo"}
            },
            ...lxTensor.Block
          },
          '---',
          {
            opcode: 'valid',
            text: 'is [TEN] a valid tensor?',
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
        blank: (generator, block) => {
          return {
            kind: 'input',
          };
        },
        blankSize: (generator, block) => {
          return {
            kind: 'input',
            shape: generator.descendInputOfBlock(block, 'SHA'),
          };
        },
        mapP: (generator, block) => {
          return {
            kind: 'input',
          };
        },
        mapV: (generator, block) => {
          return {
            kind: 'input',
          };
        },
        mapV2: (generator, block) => {
          return {
            kind: 'input',
          };
        },
        map: (generator, block) => {
          generator.script.yields = true;
          return {
            kind: 'input',
            tensor: generator.descendInputOfBlock(block, 'TEN'),
            value: generator.descendInputOfBlock(block, 'VAL'),
          };
        },
        combine: (generator, block) => {
          generator.script.yields = true;
          return {
            kind: 'input',
            tensor: generator.descendInputOfBlock(block, 'TEN'),
            tensor2: generator.descendInputOfBlock(block, 'TEN2'),
            value: generator.descendInputOfBlock(block, 'VAL'),
          };
        },
      },
      js: {
        blank: (node, compiler, imports) => {
          let source = '';
          source += `(new vm.lxTensor.Type([], true))`;
          return new imports.TypedInput(source, imports.TYPE_UNKNOWN);
        },
        blankSize: (node, compiler, imports) => {
          let source = '';
          source += `(vm.lxTensor.Type.blankSize(vm.jwArray.Type.toArray(${compiler.descendInput(node.shape).asUnknown()})))`;
          return new imports.TypedInput(source, imports.TYPE_UNKNOWN);
        },
        mapP: (node, compiler, imports) => {
          let source = '(typeof thread._lxTensorPath !== "undefined" ? vm.jwArray.Type.toArray(thread._lxTensorPath) : new vm.jwArray.Type([], true))';
          return new imports.TypedInput(source, imports.TYPE_UNKNOWN);
        },
        mapV: (node, compiler, imports) => {
          let source = '(typeof thread._lxTensorValue !== "undefined" ? thread._lxTensorValue : null)';
          return new imports.TypedInput(source, imports.TYPE_UNKNOWN);
        },
        mapV2: (node, compiler, imports) => {
          let source = '(typeof thread._lxTensorValue2 !== "undefined" ? thread._lxTensorValue2 : null)';
          return new imports.TypedInput(source, imports.TYPE_UNKNOWN);
        },
        map: (node, compiler, imports) => {
          let source = "";

          source += `vm.lxTensor.Type.toTensor(yield* (function*() {\n`;

          // input tensor
          const array = compiler.localVariables.next();
          source += `const ${array} = vm.lxTensor.Type.toTensor(${compiler.descendInput(node.tensor).asUnknown()}, true).toJSON();\n`;

          // result tensor
          const result = compiler.localVariables.next();
          source += `const ${result} = structuredClone(${array});\n`;

          // path stack (IMPORTANT)
          source += `thread._lxTensorPath ??= [];\n`;

          // recursive walker
          const walk = compiler.localVariables.next();

          const arr = compiler.localVariables.next();
          const depth = compiler.localVariables.next();
          const i = compiler.localVariables.next();
          source += `
          const ${walk} = function*(${arr}, ${depth}) {
            for (let ${i} = 0; ${i} < ${arr}.length; ${i}++) {

              thread._lxTensorPath[${depth}] = ${i} + 1;

              if (Array.isArray(${arr}[${i}])) {
                yield* ${walk}(${arr}[${i}], ${depth} + 1);
              } else {
                thread._lxTensorValue = ${arr}[${i}];
                ${arr}[${i}] = ${compiler.descendInput(node.value).asUnknown()};
              }
            }
          };`;

          source += `yield* ${walk}(${result}, 0);\n`;

          // cleanup path
          source += `thread._lxTensorPath.length = 0;\n`;

          source += `return ${result};\n`;

          source += `}()), true)\n`;

          return new imports.TypedInput(source, imports.TYPE_UNKNOWN);
        },
        combine: (node, compiler, imports) => {
          let source = "";

          source += `vm.lxTensor.Type.toTensor(yield* (function*() {\n`;

          // tensor 1 (flattened)
          const array1 = compiler.localVariables.next();
          source += `const ${array1} = vm.lxTensor.Type.toTensor(${compiler.descendInput(node.tensor).asUnknown()}, true).toJSON().flat(Infinity);\n`;

          // tensor 2 (reshaped + flattened)
          const array2 = compiler.localVariables.next();
          source += `const ${array2} = vm.lxTensor.Type.toTensor(${compiler.descendInput(node.tensor2).asUnknown()}, true).toJSON().flat(Infinity);\n`;

          // result (flat first)
          const result = compiler.localVariables.next();
          source += `const ${result} = new Array(${array1}.length);\n`;

          // loop index
          const i = compiler.localVariables.next();

          source += `for (let ${i} = 0; ${i} < ${array1}.length; ${i}++) {\n`;

          // expose scalar values
          source += `thread._lxTensorValue = ${array1}[${i}];\n`;
          source += `thread._lxTensorValue2 = ${array2}[${i}];\n`;

          // compute value
          source += `${result}[${i}] = ${compiler.descendInput(node.value).asUnknown()};\n`;

          source += `}\n`;

          // reshape back to original tensor shape
          const shapeTensor = compiler.localVariables.next();
          source += `const ${shapeTensor} = vm.lxTensor.Type.toTensor(${compiler.descendInput(node.tensor).asUnknown()}, true).shape;\n`;
          source += `return vm.lxTensor.Type.toTensor(${result}).reshape(${shapeTensor});\n`;

          source += `}()), true)\n`;

          return new imports.TypedInput(source, imports.TYPE_UNKNOWN);
        },
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

    getPath({ PAT, TEN }) {
      TEN = lxTensor.Type.toTensor(TEN);
      PAT = jwArray.Type.toArray(PAT);
      const res = TEN.getPath(PAT);
      return (res === undefined) ? '' : res;
    }
    findPath({ VAL, TEN }) {
      TEN = lxTensor.Type.toTensor(TEN);
      return TEN.findPath(VAL);
    }
    has({ TEN, VAL }) {
      TEN = lxTensor.Type.toTensor(TEN);
      return TEN.flatHas(VAL);
    }
    shape({ TEN }) {
      TEN = lxTensor.Type.toTensor(TEN);
      return new jwArray.Type(TEN.shape);
    }
    rank({ TEN }) {
      TEN = lxTensor.Type.toTensor(TEN);
      if (TEN.array == null) return 0;
      return TEN.shape.length;
    }
    scalars({ TEN }) {
      TEN = lxTensor.Type.toTensor(TEN);
      return TEN.flat(Infinity).array.length;
    }

    setPath({ PAT, TEN, VAL }) {
      TEN = lxTensor.Type.toTensor(TEN);
      PAT = jwArray.Type.toArray(PAT);
      return TEN.setPath(PAT, VAL);
    }
    reshape({ TEN, SHA }) {
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
      return TEN.transpose();
    }

    valid({ TEN }) {
      if (TEN == "" || TEN == null) return false;
      TEN = lxTensor.Type.toTensor(TEN);
      return TEN.shape.length > 0 && TEN.array.length > 0;
    }
  }
  
  Scratch.extensions.register(new Tensors());
})(Scratch);