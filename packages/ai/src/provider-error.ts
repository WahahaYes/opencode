import { Option, Schema } from "effect"
import {
  AuthenticationError,
  ContentPolicyError,
  InvalidRequestError,
  AIError,
  ProviderErrorEvent,
  ProviderInternalError,
  QuotaExceededError,
  RateLimitError,
  UnknownProviderError,
  type HttpContext,
  type HttpRateLimitDetails,
} from "./schema/index.js"

const patterns = [
  /prompt is too long/i,
  /input is too long for requested model/i,
  /exceeds the context window/i,
  /exceeds (?:the )?(?:model'?s )?maximum context length(?: of [\d,]+ tokens?|\s*\([\d,]+\))/i,
  /input token count.*exceeds the maximum/i,
  /tokens in request more than max tokens allowed/i,
  /maximum prompt length is \d+/i,
  /reduce the length of the messages/i,
  /maximum context length is \d+ tokens/i,
  /exceeds (?:the )?maximum allowed input length of [\d,]+ tokens?/i,
  /input \(\d+ tokens\) is longer than the model'?s context length \(\d+ tokens\)/i,
  /exceeds the limit of \d+/i,
  /exceeds the available context size/i,
  /greater than the context length/i,
  /context window exceeds limit/i,
  /exceeded model token limit/i,
  /context[_ ]length[_ ]exceeded/i,
  /context length is only \d+ tokens/i,
  /input length.*exceeds.*context length/i,
  /prompt too long; exceeded (?:max )?context length/i,
  /too large for model with \d+ maximum context length/i,
  /prompt has [\d,]+ tokens?, but the configured context size is [\d,]+ tokens?/i,
  /model_context_window_exceeded/i,
  /range of input length should be/i,
  /inputs`? tokens \+ `?max_new_tokens`? must be <=/i,
  /input length and `?max_tokens`? exceed context limit/i,
  /too many tokens/i,
  /token limit exceeded/i,
  /request_too_large/i,
]

const payloadPatterns = [/request entity too large/i, /payload too large/i, /request too large/i]

// Cerebras reports its per-minute token throttle as "too many tokens" with a quota code.
const TOKEN_THROTTLE_TEXT = /tokens per minute/i

const exclusions = [
  /^(throttling error|service unavailable):/i,
  /rate limit/i,
  /too many requests/i,
  TOKEN_THROTTLE_TEXT,
]

const NUMBER = String.raw`(\d[\d,]*)`

// Rejections of the requested output limit that state the largest value the provider accepts.
const outputMaximumPatterns = [
  // Anthropic
  new RegExp(String.raw`max_tokens: \d+ > ${NUMBER}, which is the maximum allowed number of output tokens`, "i"),
  // Amazon Bedrock
  new RegExp(String.raw`maximum tokens you requested exceeds the model limit of ${NUMBER}`, "i"),
  // OpenAI Chat Completions and compatible servers
  new RegExp(String.raw`supports at most ${NUMBER} completion tokens`, "i"),
  // Groq, Perplexity
  new RegExp(String.raw`max_(?:completion_)?tokens\`? must be (?:less than or equal to|at most) \`?${NUMBER}`, "i"),
  // Baseten
  new RegExp(String.raw`max_tokens \(\d+\): input should be less than or equal to ${NUMBER}`, "i"),
  // DeepSeek, Alibaba
  new RegExp(String.raw`range of max_tokens (?:is|should be) \[\s*\d+\s*,\s*${NUMBER}\s*\]`, "i"),
  // Z.AI
  new RegExp(String.raw`max_tokens parameter is illegal[^\[]*\[\s*\d+\s*,\s*${NUMBER}\s*\]`, "i"),
  // MiniMax
  new RegExp(String.raw`does not support max tokens (?:>|\\u003e) ${NUMBER}`, "i"),
]

// Rejections of input plus requested output above the context window. Each reads the window and the input size.
const contextSumPatterns: ReadonlyArray<{ pattern: RegExp; read: (match: RegExpExecArray) => [number, number] }> = [
  // OpenAI-compatible servers and OpenRouter list the requested output last, after every input category.
  {
    pattern: new RegExp(
      String.raw`maximum context length is ${NUMBER} tokens\. However, you requested (?:about )?${NUMBER} tokens \(.*?${NUMBER} in the (?:completion|output)\)`,
      "i",
    ),
    read: (match) => [count(match[1]), count(match[2]) - count(match[3])],
  },
  // vLLM
  {
    pattern: new RegExp(
      String.raw`maximum context length is ${NUMBER} tokens and your request has ${NUMBER} input tokens`,
      "i",
    ),
    read: (match) => [count(match[1]), count(match[2])],
  },
  // Together AI
  {
    pattern: new RegExp(
      String.raw`inputs\`? tokens \+ \`?max_new_tokens\`? must be <= ${NUMBER}\. Given: ${NUMBER} \`?inputs`,
      "i",
    ),
    read: (match) => [count(match[1]), count(match[2])],
  },
  // Anthropic
  {
    pattern: new RegExp(
      String.raw`input length and \`?max_tokens\`? exceed context limit: ${NUMBER} \+ \d[\d,]* > ${NUMBER}`,
      "i",
    ),
    read: (match) => [count(match[2]), count(match[1])],
  },
]

function count(value: string) {
  return Number(value.replaceAll(",", ""))
}

/**
 * The largest output limit a provider accepts after rejecting the requested one, stated directly or derived from the
 * context window and input size it reports. `undefined` when the input alone fills the window, which is an overflow.
 */
const acceptedOutputLimit = (message: string) => {
  const stated = outputMaximumPatterns.map((pattern) => pattern.exec(message)?.[1]).find((value) => value !== undefined)
  if (stated !== undefined) return count(stated)
  const sum = contextSumPatterns
    .map((item) => {
      const match = item.pattern.exec(message)
      return match ? item.read(match) : undefined
    })
    .find((value) => value !== undefined)
  if (!sum) return undefined
  const accepted = sum[0] - sum[1]
  return accepted > 0 ? accepted : undefined
}

export const isContextOverflow = (message: string) =>
  !exclusions.some((pattern) => pattern.test(message)) &&
  acceptedOutputLimit(message) === undefined &&
  (patterns.some((pattern) => pattern.test(message)) || /^4(?:00|13)\s*(status code)?\s*\(no body\)/i.test(message))

export const isPayloadTooLarge = (message: string) => payloadPatterns.some((pattern) => pattern.test(message))

export const isContextOverflowFailure = (failure: unknown) =>
  failure instanceof AIError
    ? failure.reason._tag === "InvalidRequest" && failure.reason.classification === "context-overflow"
    : Schema.is(ProviderErrorEvent)(failure) && failure.classification === "context-overflow"

/** The `maxTokens` a provider accepts after rejecting the requested output limit; `undefined` for any other failure. */
export const acceptedMaxTokens = (failure: unknown) =>
  failure instanceof AIError &&
  failure.reason._tag === "InvalidRequest" &&
  failure.reason.classification === "output-limit"
    ? failure.reason.acceptedMaxTokens
    : undefined

const decodeJson = Schema.decodeUnknownOption(Schema.fromJsonString(Schema.Unknown))
// OpenCode Zen reports account caps as typed 429/402 errors that are not throttles.
const QUOTA_CODES = new Set([
  "insufficient_quota",
  "usage_not_included",
  "billing_error",
  "gousagelimiterror",
  "freeusagelimiterror",
  "creditlimitexceeded",
])
const AUTH_CODES = new Set(["authentication_error", "permission_error"])
const SERVER_CODES = new Set([
  "api_error",
  "internal_error",
  "internalserverexception",
  "modelstreamerrorexception",
  "overloaded_error",
  "server_error",
  "server_is_overloaded",
  "slow_down",
  "serviceunavailableexception",
])
const INVALID_REQUEST_CODES = new Set(["invalid_prompt", "invalid_request_error", "validationexception"])
// Azure OpenAI reports `content_filter` with `innererror.code` ResponsibleAIPolicyViolation.
// OpenRouter tags provider failures with a typed `error_type`; its Responses skin also
// emits `image_content_policy_violation` as the native code.
const CONTENT_POLICY_CODES = new Set([
  "content_filter",
  "content_moderation",
  "responsibleaipolicyviolation",
  "content_policy_violation",
  "image_content_policy_violation",
  "refusal",
])
// OpenCode Zen replaces upstream codes outside its allow-list but keeps the original
// as a `[code]` label at the start of the rewritten message.
const GATEWAY_CODE_LABEL = /^[^:\n]+: \[([A-Za-z0-9_.-]+)\]/
const RATE_LIMIT_TEXT = /rate increased too quickly|rate[-_\s]?limit|too[_\s]?many[_\s]?requests/i
// Only consulted on 429, where throttles and account caps share a status.
const QUOTA_TEXT = /insufficient[-_\s]?quota|quota[-_\s]?exceeded|budget exceeded|usage limit/i
// Policy rejections without a dedicated code, matched against the provider's own
// explanation only. OpenAI reuses `invalid_prompt` for usage-policy rejections while
// Bedrock Mantle reuses it for schema validation; Anthropic reports blocked output
// under `invalid_request_error`.
const CONTENT_POLICY_TEXT =
  /violating our usage policy|blocked by content filtering policy|content[-_\s]?policy|rejected as a result of our safety system/i
const SERVER_ERROR_TEXT =
  /\b(?:try again|(?:please |you can )?retry (?:the |this |your )?request|try (?:the |this |your )?request again|(?:currently |temporarily )?at capacity|overloaded|temporarily unavailable|service[-_\s]?unavailable|(?:server|internal)[-_\s]?error|server (?:is )?busy|provider returned (?:an )?error|resource[-_\s]?exhausted|upstream (?:connect|connection|request)|request buffer limit while retrying upstream)\b/i

export interface ProviderFailure {
  readonly message: string
  readonly status?: number | undefined
  // Raw wire payload, scanned for failure signals (codes, overflow phrases)
  // that the summary message does not carry. Not shown to users.
  readonly rawBody?: string | undefined
  // Some SDKs supply parsed error data separately from the original response text.
  readonly data?: unknown
  readonly http?: HttpContext | undefined
  readonly cause?: unknown
  readonly retryAfterMs?: number | undefined
  readonly rateLimit?: HttpRateLimitDetails | undefined
}

// Classification records affirmative evidence about a failure. Deterministic
// failures need positive identification (a 4xx status, quota/auth/policy
// signals); anything unrecognized stays UnknownProvider, which the session
// retry policy treats as retry-eligible because transient failures arrive in
// unpredictable shapes while deterministic rejections almost always carry a
// status or known code.
export function classifyProviderFailure(input: ProviderFailure): AIError["reason"] {
  const details = { message: input.message, body: input.rawBody, http: input.http, cause: input.cause }
  const body = input.rawBody ?? ""
  const codes = [
    ...providerCodes(input.data),
    ...providerCodes(body),
    ...providerCodes(input.message),
    ...(GATEWAY_CODE_LABEL.exec(input.message)?.slice(1) ?? []),
  ].map((code) => code.toLowerCase())
  // Scan the raw payload too so signals missing from the summary message
  // (e.g. overflow phrases nested in a JSON error body) still classify.
  const text = [input.message, body].filter((value) => value.length > 0).join("\n")
  const clientScoped = input.status === undefined || (input.status >= 400 && input.status < 500)

  const acceptedMaxTokens = clientScoped ? acceptedOutputLimit(text) : undefined
  if (acceptedMaxTokens !== undefined)
    return new InvalidRequestError({ ...details, classification: "output-limit", acceptedMaxTokens })
  if (
    clientScoped &&
    (codes.includes("context_length_exceeded") ||
      codes.includes("model_context_window_exceeded") ||
      codes.includes("request_too_large") ||
      isContextOverflow(text))
  )
    return new InvalidRequestError({ ...details, classification: "context-overflow" })
  if (input.status === 413 || isPayloadTooLarge(text))
    return new InvalidRequestError({ ...details, classification: "payload-too-large" })
  if (codes.some((code) => CONTENT_POLICY_CODES.has(code)) || (clientScoped && CONTENT_POLICY_TEXT.test(input.message)))
    return new ContentPolicyError(details)
  if (
    input.status === 402 ||
    codes.some((code) => QUOTA_CODES.has(code)) ||
    (input.status === 429 && QUOTA_TEXT.test(text) && !TOKEN_THROTTLE_TEXT.test(text))
  )
    return new QuotaExceededError(details)
  if (input.status === 401 || input.status === 403 || codes.some((code) => AUTH_CODES.has(code)))
    return new AuthenticationError(details)
  if (
    input.status === 429 ||
    codes.some(
      (code) => code.includes("rate_limit") || code === "too_many_requests" || code === "throttlingexception",
    ) ||
    RATE_LIMIT_TEXT.test(text)
  )
    return new RateLimitError({
      ...details,
      retryAfterMs: input.retryAfterMs,
      rateLimit: input.rateLimit,
    })
  if (
    input.status === 408 ||
    input.status === 409 ||
    (input.status !== undefined && input.status >= 500) ||
    // Server codes and phrasing only decide when no HTTP status contradicts them:
    // gateways such as OpenCode Zen substitute `server_error` for codes they do
    // not forward, so a 4xx with a server code is still a rejected request.
    ((input.status === undefined || input.status < 400) &&
      ((!codes.some((code) => INVALID_REQUEST_CODES.has(code)) && SERVER_ERROR_TEXT.test(text)) ||
        codes.some((code) => SERVER_CODES.has(code) || code.includes("exhausted") || code.includes("unavailable"))))
  )
    return new ProviderInternalError({
      ...details,
      retryAfterMs: input.retryAfterMs,
    })
  if (codes.some((code) => INVALID_REQUEST_CODES.has(code))) return new InvalidRequestError(details)
  // Any remaining 4xx is a deterministic rejection of this request.
  if (input.status !== undefined && input.status >= 400 && input.status < 500) return new InvalidRequestError(details)
  return new UnknownProviderError(details)
}

function providerCodes(value: unknown) {
  const decoded = typeof value === "string" ? Option.getOrUndefined(decodeJson(value)) : value
  if (!isRecord(decoded)) return []
  const error = isRecord(decoded.error) ? decoded.error : undefined
  const inner = error && isRecord(error.innererror) ? error.innererror : undefined
  const metadata = error && isRecord(error.metadata) ? error.metadata : undefined
  const response = isRecord(decoded.response) ? decoded.response : undefined
  const responseError = response && isRecord(response.error) ? response.error : undefined
  const exception = isRecord(decoded.exception) ? decoded.exception : undefined
  return [
    decoded.code,
    // Stability's `{ id, name, errors }` bodies carry the code in `name`.
    Array.isArray(decoded.errors) ? decoded.name : undefined,
    decoded.error_type,
    error?.code,
    error?.type,
    error?.status,
    error?.error_type,
    inner?.code,
    metadata?.error_type,
    responseError?.code,
    response?.error_type,
    exception?.type,
  ].filter((value): value is string => typeof value === "string")
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null
}
