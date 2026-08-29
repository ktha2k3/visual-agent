import { createClient } from '@supabase/supabase-js'

const url = import.meta.env.VITE_SUPABASE_URL
const key = import.meta.env.VITE_SUPABASE_PUBLISHABLE_KEY

export const supabase = url && key
  ? createClient(url, key, {
      auth: { persistSession: true, autoRefreshToken: true, detectSessionInUrl: true },
    })
  : null

export async function authHeaders() {
  const token = await accessToken()
  return token ? { Authorization: `Bearer ${token}` } : {}
}

export async function accessToken() {
  if (!supabase) return null
  const { data } = await supabase.auth.getSession()
  return data.session?.access_token || null
}
