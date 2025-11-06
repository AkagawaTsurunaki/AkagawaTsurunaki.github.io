export function withTiming<F extends (...args: any[]) => any>(fn: F): F {
  return function (this: any, ...args: Parameters<F>): ReturnType<F> {
    const start = performance.now()
    const result = fn.apply(this, args)

    if (result instanceof Promise) {
      return result.then((data: any) => {
        const end = performance.now()
        console.log(`${fn.name || 'Function'} used ${(end - start).toFixed(3)}ms`)
        return data
      }) as ReturnType<F>
    } else {
      const end = performance.now()
      console.log(`${fn.name || 'Function'} used ${(end - start).toFixed(3)}ms`)
      return result
    }
  } as F
}

// Example:
// const expensiveCalculation = withTiming(function (n: number): number {
//   let sum = 0
//   for (let i = 0; i < n; i++) {
//     sum += i
//   }
//   return sum
// })
// expensiveCalculation()
