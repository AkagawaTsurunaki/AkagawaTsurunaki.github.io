export async function readFileText(path: string): Promise<string | null> {
  const res = await fetch(path)
  if (!res.ok) return null
  const text = await res.text()
  return text
}

export async function readFileJson(path: string): Promise<any> {
  const jsonStr = await readFileText(path)
  if (!jsonStr) return null
  return JSON.parse(jsonStr)
}
