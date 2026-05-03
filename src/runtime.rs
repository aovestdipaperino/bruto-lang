/// Bruto runtime — shared LLVM IR functions for memory, I/O, and capture.
///
/// Call `emit_runtime(context, module)` to inject all `bruto_*` functions
/// into an LLVM module.  Each is a thin wrapper around libc.
use inkwell::AddressSpace;
use inkwell::context::Context;
use inkwell::module::Module;

/// Emit all bruto runtime function definitions into `module`.
///
/// After this call the module contains callable functions:
///   bruto_alloc, bruto_free,
///   bruto_write_int, bruto_write_str, bruto_write_bool, bruto_writeln,
///   bruto_read_int,
///   bruto_capture_open, bruto_capture_write_int, bruto_capture_write_str,
///   bruto_capture_write_bool, bruto_capture_writeln, bruto_capture_close
pub fn emit_runtime<'ctx>(context: &'ctx Context, module: &Module<'ctx>) {
    let i32_ty = context.i32_type();
    let i64_ty = context.i64_type();
    let i1_ty = context.bool_type();
    let void_ty = context.void_type();
    let ptr_ty = context.ptr_type(AddressSpace::default());

    // ── libc externs ────────────────────────────────────────
    let printf_ty = i32_ty.fn_type(&[ptr_ty.into()], true);
    module.add_function("printf", printf_ty, None);

    let putchar_ty = i32_ty.fn_type(&[i32_ty.into()], false);
    module.add_function("putchar", putchar_ty, None);

    let scanf_ty = i32_ty.fn_type(&[ptr_ty.into()], true);
    module.add_function("scanf", scanf_ty, None);

    let malloc_ty = ptr_ty.fn_type(&[i64_ty.into()], false);
    module.add_function("malloc", malloc_ty, None);

    let free_ty = void_ty.fn_type(&[ptr_ty.into()], false);
    module.add_function("free", free_ty, None);

    let fopen_ty = ptr_ty.fn_type(&[ptr_ty.into(), ptr_ty.into()], false);
    module.add_function("fopen", fopen_ty, None);

    let fclose_ty = i32_ty.fn_type(&[ptr_ty.into()], false);
    module.add_function("fclose", fclose_ty, None);

    let fprintf_ty = i32_ty.fn_type(&[ptr_ty.into(), ptr_ty.into()], true);
    module.add_function("fprintf", fprintf_ty, None);

    let fflush_ty = i32_ty.fn_type(&[ptr_ty.into()], false);
    module.add_function("fflush", fflush_ty, None);

    // ── global: capture FILE* ───────────────────────────────
    let capture_global = module.add_global(ptr_ty, None, "_bruto_capture_fp");
    capture_global.set_initializer(&ptr_ty.const_null());

    // ── global: ioresult error code ─────────────────────────
    let io_global = module.add_global(i32_ty, None, "_bruto_ioresult");
    io_global.set_initializer(&i32_ty.const_int(0, false));

    // ── bruto_alloc(i64) -> ptr ─────────────────────────────
    {
        let fn_ty = ptr_ty.fn_type(&[i64_ty.into()], false);
        let func = module.add_function("bruto_alloc", fn_ty, None);
        let bb = context.append_basic_block(func, "entry");
        let b = context.create_builder();
        b.position_at_end(bb);
        let size = func.get_first_param().unwrap().into_int_value();
        let malloc = module.get_function("malloc").unwrap();
        let ptr = b
            .build_call(malloc, &[size.into()], "ptr")
            .unwrap()
            .try_as_basic_value()
            .basic()
            .unwrap();
        b.build_return(Some(&ptr)).unwrap();
    }

    // ── bruto_free(ptr) ─────────────────────────────────────
    {
        let fn_ty = void_ty.fn_type(&[ptr_ty.into()], false);
        let func = module.add_function("bruto_free", fn_ty, None);
        let bb = context.append_basic_block(func, "entry");
        let b = context.create_builder();
        b.position_at_end(bb);
        let p = func.get_first_param().unwrap().into_pointer_value();
        let free = module.get_function("free").unwrap();
        b.build_call(free, &[p.into()], "").unwrap();
        b.build_return(None).unwrap();
    }

    // ── bruto_write_int(i64) ────────────────────────────────
    {
        let fn_ty = void_ty.fn_type(&[i64_ty.into()], false);
        let func = module.add_function("bruto_write_int", fn_ty, None);
        let bb = context.append_basic_block(func, "entry");
        let b = context.create_builder();
        b.position_at_end(bb);
        let val = func.get_first_param().unwrap().into_int_value();
        let printf = module.get_function("printf").unwrap();
        let fmt = b.build_global_string_ptr("%lld", "fmt_int").unwrap();
        b.build_call(printf, &[fmt.as_pointer_value().into(), val.into()], "")
            .unwrap();
        b.build_return(None).unwrap();
    }

    // ── bruto_write_str(ptr) ────────────────────────────────
    {
        let fn_ty = void_ty.fn_type(&[ptr_ty.into()], false);
        let func = module.add_function("bruto_write_str", fn_ty, None);
        let bb = context.append_basic_block(func, "entry");
        let b = context.create_builder();
        b.position_at_end(bb);
        let s = func.get_first_param().unwrap().into_pointer_value();
        let printf = module.get_function("printf").unwrap();
        let fmt = b.build_global_string_ptr("%s", "fmt_str").unwrap();
        b.build_call(printf, &[fmt.as_pointer_value().into(), s.into()], "")
            .unwrap();
        b.build_return(None).unwrap();
    }

    // ── bruto_write_bool(i1) ────────────────────────────────
    {
        let fn_ty = void_ty.fn_type(&[i1_ty.into()], false);
        let func = module.add_function("bruto_write_bool", fn_ty, None);
        let bb = context.append_basic_block(func, "entry");
        let b = context.create_builder();
        b.position_at_end(bb);
        let val = func.get_first_param().unwrap().into_int_value();
        let true_s = b.build_global_string_ptr("true", "true_s").unwrap();
        let false_s = b.build_global_string_ptr("false", "false_s").unwrap();
        let sel = b
            .build_select(
                val,
                true_s.as_pointer_value(),
                false_s.as_pointer_value(),
                "bs",
            )
            .unwrap();
        let printf = module.get_function("printf").unwrap();
        let fmt = b.build_global_string_ptr("%s", "fmt_bs").unwrap();
        b.build_call(printf, &[fmt.as_pointer_value().into(), sel.into()], "")
            .unwrap();
        b.build_return(None).unwrap();
    }

    // ── bruto_write_real(f64) ──────────────────────────────
    {
        let f64_ty = context.f64_type();
        let fn_ty = void_ty.fn_type(&[f64_ty.into()], false);
        let func = module.add_function("bruto_write_real", fn_ty, None);
        let bb = context.append_basic_block(func, "entry");
        let b = context.create_builder();
        b.position_at_end(bb);
        let val = func.get_first_param().unwrap().into_float_value();
        let printf = module.get_function("printf").unwrap();
        let fmt = b.build_global_string_ptr("%.10g", "fmt_real").unwrap();
        b.build_call(printf, &[fmt.as_pointer_value().into(), val.into()], "")
            .unwrap();
        b.build_return(None).unwrap();
    }

    // ── bruto_write_char(i8) ────────────────────────────────
    {
        let i8_ty = context.i8_type();
        let fn_ty = void_ty.fn_type(&[i8_ty.into()], false);
        let func = module.add_function("bruto_write_char", fn_ty, None);
        let bb = context.append_basic_block(func, "entry");
        let b = context.create_builder();
        b.position_at_end(bb);
        let val = func.get_first_param().unwrap().into_int_value();
        let putchar = module.get_function("putchar").unwrap();
        let ext = b.build_int_z_extend(val, i32_ty, "zext").unwrap();
        b.build_call(putchar, &[ext.into()], "").unwrap();
        b.build_return(None).unwrap();
    }

    // ── bruto_writeln() ─────────────────────────────────────
    {
        let fn_ty = void_ty.fn_type(&[], false);
        let func = module.add_function("bruto_writeln", fn_ty, None);
        let bb = context.append_basic_block(func, "entry");
        let b = context.create_builder();
        b.position_at_end(bb);
        let putchar = module.get_function("putchar").unwrap();
        b.build_call(putchar, &[i32_ty.const_int(10, false).into()], "")
            .unwrap();
        b.build_return(None).unwrap();
    }

    // ── bruto_read_int() -> i64 ─────────────────────────────
    {
        let fn_ty = i64_ty.fn_type(&[], false);
        let func = module.add_function("bruto_read_int", fn_ty, None);
        let bb = context.append_basic_block(func, "entry");
        let b = context.create_builder();
        b.position_at_end(bb);
        let tmp = b.build_alloca(i64_ty, "tmp").unwrap();
        b.build_store(tmp, i64_ty.const_int(0, false)).unwrap();
        let scanf = module.get_function("scanf").unwrap();
        let fmt = b.build_global_string_ptr("%lld", "fmt_scan").unwrap();
        b.build_call(scanf, &[fmt.as_pointer_value().into(), tmp.into()], "")
            .unwrap();
        let val = b.build_load(i64_ty, tmp, "val").unwrap();
        b.build_return(Some(&val)).unwrap();
    }

    // ── bruto_capture_open(ptr path) ────────────────────────
    {
        let fn_ty = void_ty.fn_type(&[ptr_ty.into()], false);
        let func = module.add_function("bruto_capture_open", fn_ty, None);
        let bb = context.append_basic_block(func, "entry");
        let b = context.create_builder();
        b.position_at_end(bb);
        let path = func.get_first_param().unwrap().into_pointer_value();
        let fopen = module.get_function("fopen").unwrap();
        let mode = b.build_global_string_ptr("w", "cap_mode").unwrap();
        let fp = b
            .build_call(fopen, &[path.into(), mode.as_pointer_value().into()], "fp")
            .unwrap()
            .try_as_basic_value()
            .basic()
            .unwrap();
        let g = module.get_global("_bruto_capture_fp").unwrap();
        b.build_store(g.as_pointer_value(), fp).unwrap();
        b.build_return(None).unwrap();
    }

    // helper: load capture fp, return (builder, fp_val, is_null_bb, write_bb)
    // We'll inline the pattern for each capture_write variant instead.

    // ── bruto_capture_write_int(i64) ────────────────────────
    emit_capture_write_fn(
        context,
        module,
        "bruto_capture_write_int",
        &[i64_ty.into()],
        "%lld",
    );

    // ── bruto_capture_write_str(ptr) ────────────────────────
    emit_capture_write_fn(
        context,
        module,
        "bruto_capture_write_str",
        &[ptr_ty.into()],
        "%s",
    );

    // ── bruto_capture_write_bool(i1) — needs special handling
    {
        let fn_ty = void_ty.fn_type(&[i1_ty.into()], false);
        let func = module.add_function("bruto_capture_write_bool", fn_ty, None);
        let bb = context.append_basic_block(func, "entry");
        let b = context.create_builder();
        b.position_at_end(bb);

        let g = module.get_global("_bruto_capture_fp").unwrap();
        let fp = b
            .build_load(ptr_ty, g.as_pointer_value(), "fp")
            .unwrap()
            .into_pointer_value();
        let is_null = b.build_is_null(fp, "null").unwrap();

        let write_bb = context.append_basic_block(func, "write");
        let end_bb = context.append_basic_block(func, "end");
        b.build_conditional_branch(is_null, end_bb, write_bb)
            .unwrap();

        b.position_at_end(write_bb);
        let val = func.get_first_param().unwrap().into_int_value();
        let true_s = b.build_global_string_ptr("true", "cap_true").unwrap();
        let false_s = b.build_global_string_ptr("false", "cap_false").unwrap();
        let sel = b
            .build_select(
                val,
                true_s.as_pointer_value(),
                false_s.as_pointer_value(),
                "bs",
            )
            .unwrap();
        let fprintf = module.get_function("fprintf").unwrap();
        let fmt = b.build_global_string_ptr("%s", "cap_fmt_bs").unwrap();
        b.build_call(
            fprintf,
            &[fp.into(), fmt.as_pointer_value().into(), sel.into()],
            "",
        )
        .unwrap();
        let fflush = module.get_function("fflush").unwrap();
        b.build_call(fflush, &[fp.into()], "").unwrap();
        b.build_unconditional_branch(end_bb).unwrap();

        b.position_at_end(end_bb);
        b.build_return(None).unwrap();
    }

    // ── bruto_capture_write_real(f64) ─────────────────────
    emit_capture_write_fn(
        context,
        module,
        "bruto_capture_write_real",
        &[context.f64_type().into()],
        "%.10g",
    );

    // ── bruto_capture_write_char(i8) ────────────────────
    {
        let i8_ty = context.i8_type();
        let fn_ty = void_ty.fn_type(&[i8_ty.into()], false);
        let func = module.add_function("bruto_capture_write_char", fn_ty, None);
        let bb = context.append_basic_block(func, "entry");
        let b = context.create_builder();
        b.position_at_end(bb);

        let g = module.get_global("_bruto_capture_fp").unwrap();
        let fp = b
            .build_load(ptr_ty, g.as_pointer_value(), "fp")
            .unwrap()
            .into_pointer_value();
        let is_null = b.build_is_null(fp, "null").unwrap();

        let write_bb = context.append_basic_block(func, "write");
        let end_bb = context.append_basic_block(func, "end");
        b.build_conditional_branch(is_null, end_bb, write_bb)
            .unwrap();

        b.position_at_end(write_bb);
        let val = func.get_first_param().unwrap().into_int_value();
        let fprintf = module.get_function("fprintf").unwrap();
        let fmt = b.build_global_string_ptr("%c", "cap_fmt_ch").unwrap();
        let ext = b.build_int_z_extend(val, i32_ty, "zext").unwrap();
        b.build_call(
            fprintf,
            &[fp.into(), fmt.as_pointer_value().into(), ext.into()],
            "",
        )
        .unwrap();
        let fflush = module.get_function("fflush").unwrap();
        b.build_call(fflush, &[fp.into()], "").unwrap();
        b.build_unconditional_branch(end_bb).unwrap();

        b.position_at_end(end_bb);
        b.build_return(None).unwrap();
    }

    // ── bruto_capture_writeln() ─────────────────────────────
    {
        let fn_ty = void_ty.fn_type(&[], false);
        let func = module.add_function("bruto_capture_writeln", fn_ty, None);
        let bb = context.append_basic_block(func, "entry");
        let b = context.create_builder();
        b.position_at_end(bb);

        let g = module.get_global("_bruto_capture_fp").unwrap();
        let fp = b
            .build_load(ptr_ty, g.as_pointer_value(), "fp")
            .unwrap()
            .into_pointer_value();
        let is_null = b.build_is_null(fp, "null").unwrap();

        let write_bb = context.append_basic_block(func, "write");
        let end_bb = context.append_basic_block(func, "end");
        b.build_conditional_branch(is_null, end_bb, write_bb)
            .unwrap();

        b.position_at_end(write_bb);
        let fprintf = module.get_function("fprintf").unwrap();
        let fmt = b.build_global_string_ptr("\n", "cap_nl").unwrap();
        b.build_call(fprintf, &[fp.into(), fmt.as_pointer_value().into()], "")
            .unwrap();
        let fflush = module.get_function("fflush").unwrap();
        b.build_call(fflush, &[fp.into()], "").unwrap();
        b.build_unconditional_branch(end_bb).unwrap();

        b.position_at_end(end_bb);
        b.build_return(None).unwrap();
    }

    // ── bruto_capture_close() ───────────────────────────────
    {
        let fn_ty = void_ty.fn_type(&[], false);
        let func = module.add_function("bruto_capture_close", fn_ty, None);
        let bb = context.append_basic_block(func, "entry");
        let b = context.create_builder();
        b.position_at_end(bb);

        let g = module.get_global("_bruto_capture_fp").unwrap();
        let fp = b
            .build_load(ptr_ty, g.as_pointer_value(), "fp")
            .unwrap()
            .into_pointer_value();
        let is_null = b.build_is_null(fp, "null").unwrap();

        let close_bb = context.append_basic_block(func, "close");
        let end_bb = context.append_basic_block(func, "end");
        b.build_conditional_branch(is_null, end_bb, close_bb)
            .unwrap();

        b.position_at_end(close_bb);
        let fclose = module.get_function("fclose").unwrap();
        b.build_call(fclose, &[fp.into()], "").unwrap();
        b.build_store(g.as_pointer_value(), ptr_ty.const_null())
            .unwrap();
        b.build_unconditional_branch(end_bb).unwrap();

        b.position_at_end(end_bb);
        b.build_return(None).unwrap();
    }

    // ── String operations ───────────────────────────────

    // libc string externs
    let strlen_ty = i64_ty.fn_type(&[ptr_ty.into()], false);
    module.add_function("strlen", strlen_ty, None);

    let strcpy_ty = ptr_ty.fn_type(&[ptr_ty.into(), ptr_ty.into()], false);
    module.add_function("strcpy", strcpy_ty, None);

    let strcat_ty = ptr_ty.fn_type(&[ptr_ty.into(), ptr_ty.into()], false);
    module.add_function("strcat", strcat_ty, None);

    let strcmp_ty = i32_ty.fn_type(&[ptr_ty.into(), ptr_ty.into()], false);
    module.add_function("strcmp", strcmp_ty, None);

    // bruto_str_concat(a: ptr, b: ptr) -> ptr  (new heap string = a + b)
    {
        let fn_ty = ptr_ty.fn_type(&[ptr_ty.into(), ptr_ty.into()], false);
        let func = module.add_function("bruto_str_concat", fn_ty, None);
        let bb = context.append_basic_block(func, "entry");
        let b = context.create_builder();
        b.position_at_end(bb);
        let a = func.get_nth_param(0).unwrap().into_pointer_value();
        let bv = func.get_nth_param(1).unwrap().into_pointer_value();
        let strlen = module.get_function("strlen").unwrap();
        let len_a = b
            .build_call(strlen, &[a.into()], "la")
            .unwrap()
            .try_as_basic_value()
            .basic()
            .unwrap()
            .into_int_value();
        let len_b = b
            .build_call(strlen, &[bv.into()], "lb")
            .unwrap()
            .try_as_basic_value()
            .basic()
            .unwrap()
            .into_int_value();
        let total = b.build_int_add(len_a, len_b, "tot").unwrap();
        let total_plus1 = b
            .build_int_add(total, i64_ty.const_int(1, false), "tot1")
            .unwrap();
        let malloc = module.get_function("malloc").unwrap();
        let buf = b
            .build_call(malloc, &[total_plus1.into()], "buf")
            .unwrap()
            .try_as_basic_value()
            .basic()
            .unwrap();
        let strcpy = module.get_function("strcpy").unwrap();
        b.build_call(strcpy, &[buf.into(), a.into()], "").unwrap();
        let strcat = module.get_function("strcat").unwrap();
        b.build_call(strcat, &[buf.into(), bv.into()], "").unwrap();
        b.build_return(Some(&buf)).unwrap();
    }

    // bruto_str_length(s: ptr) -> i64
    {
        let fn_ty = i64_ty.fn_type(&[ptr_ty.into()], false);
        let func = module.add_function("bruto_str_length", fn_ty, None);
        let bb = context.append_basic_block(func, "entry");
        let b = context.create_builder();
        b.position_at_end(bb);
        let s = func.get_first_param().unwrap().into_pointer_value();
        let strlen = module.get_function("strlen").unwrap();
        let len = b
            .build_call(strlen, &[s.into()], "len")
            .unwrap()
            .try_as_basic_value()
            .basic()
            .unwrap();
        b.build_return(Some(&len)).unwrap();
    }

    // bruto_str_compare(a: ptr, b: ptr) -> i32  (strcmp wrapper)
    {
        let fn_ty = i32_ty.fn_type(&[ptr_ty.into(), ptr_ty.into()], false);
        let func = module.add_function("bruto_str_compare", fn_ty, None);
        let bb = context.append_basic_block(func, "entry");
        let b = context.create_builder();
        b.position_at_end(bb);
        let a = func.get_nth_param(0).unwrap().into_pointer_value();
        let bv = func.get_nth_param(1).unwrap().into_pointer_value();
        let strcmp = module.get_function("strcmp").unwrap();
        let r = b
            .build_call(strcmp, &[a.into(), bv.into()], "cmp")
            .unwrap()
            .try_as_basic_value()
            .basic()
            .unwrap();
        b.build_return(Some(&r)).unwrap();
    }

    // ── libc externs for string ops ────────────────────────
    let memcpy_ty = ptr_ty.fn_type(&[ptr_ty.into(), ptr_ty.into(), i64_ty.into()], false);
    module.add_function("memcpy", memcpy_ty, None);

    let snprintf_ty = i32_ty.fn_type(&[ptr_ty.into(), i64_ty.into(), ptr_ty.into()], true);
    module.add_function("snprintf", snprintf_ty, None);

    let strstr_ty = ptr_ty.fn_type(&[ptr_ty.into(), ptr_ty.into()], false);
    module.add_function("strstr", strstr_ty, None);

    let atoll_ty = i64_ty.fn_type(&[ptr_ty.into()], false);
    module.add_function("atoll", atoll_ty, None);

    // bruto_str_copy(s: ptr, index: i64, count: i64) -> ptr
    // copy(s, index, count) — 1-based index
    {
        let fn_ty = ptr_ty.fn_type(&[ptr_ty.into(), i64_ty.into(), i64_ty.into()], false);
        let func = module.add_function("bruto_str_copy", fn_ty, None);
        let bb = context.append_basic_block(func, "entry");
        let b = context.create_builder();
        b.position_at_end(bb);
        let s = func.get_nth_param(0).unwrap().into_pointer_value();
        let index = func.get_nth_param(1).unwrap().into_int_value();
        let count = func.get_nth_param(2).unwrap().into_int_value();
        // start = index - 1
        let start = b
            .build_int_sub(index, i64_ty.const_int(1, false), "start")
            .unwrap();
        // allocate count+1 bytes
        let buf_size = b
            .build_int_add(count, i64_ty.const_int(1, false), "bsz")
            .unwrap();
        let malloc = module.get_function("malloc").unwrap();
        let buf = b
            .build_call(malloc, &[buf_size.into()], "buf")
            .unwrap()
            .try_as_basic_value()
            .basic()
            .unwrap()
            .into_pointer_value();
        // src = s + start
        let src = unsafe {
            b.build_in_bounds_gep(context.i8_type(), s, &[start], "src")
                .unwrap()
        };
        let memcpy = module.get_function("memcpy").unwrap();
        b.build_call(memcpy, &[buf.into(), src.into(), count.into()], "")
            .unwrap();
        // null-terminate: buf[count] = 0
        let end = unsafe {
            b.build_in_bounds_gep(context.i8_type(), buf, &[count], "end")
                .unwrap()
        };
        b.build_store(end, context.i8_type().const_int(0, false))
            .unwrap();
        b.build_return(Some(&buf)).unwrap();
    }

    // bruto_str_pos(substr: ptr, s: ptr) -> i64
    // pos(substr, s) — returns 1-based position, 0 if not found
    {
        let fn_ty = i64_ty.fn_type(&[ptr_ty.into(), ptr_ty.into()], false);
        let func = module.add_function("bruto_str_pos", fn_ty, None);
        let bb = context.append_basic_block(func, "entry");
        let b = context.create_builder();
        b.position_at_end(bb);
        let substr = func.get_nth_param(0).unwrap().into_pointer_value();
        let s = func.get_nth_param(1).unwrap().into_pointer_value();
        let strstr = module.get_function("strstr").unwrap();
        let found = b
            .build_call(strstr, &[s.into(), substr.into()], "found")
            .unwrap()
            .try_as_basic_value()
            .basic()
            .unwrap()
            .into_pointer_value();
        let is_null = b.build_is_null(found, "null").unwrap();
        let found_bb = context.append_basic_block(func, "found");
        let not_found_bb = context.append_basic_block(func, "not_found");
        b.build_conditional_branch(is_null, not_found_bb, found_bb)
            .unwrap();

        b.position_at_end(found_bb);
        let s_int = b.build_ptr_to_int(s, i64_ty, "si").unwrap();
        let f_int = b.build_ptr_to_int(found, i64_ty, "fi").unwrap();
        let diff = b.build_int_sub(f_int, s_int, "diff").unwrap();
        let pos = b
            .build_int_add(diff, i64_ty.const_int(1, false), "pos")
            .unwrap(); // 1-based
        b.build_return(Some(&pos)).unwrap();

        b.position_at_end(not_found_bb);
        b.build_return(Some(&i64_ty.const_int(0, false))).unwrap();
    }

    // bruto_str_delete(s: ptr, index: i64, count: i64) -> ptr
    {
        let fn_ty = ptr_ty.fn_type(&[ptr_ty.into(), i64_ty.into(), i64_ty.into()], false);
        let func = module.add_function("bruto_str_delete", fn_ty, None);
        let bb = context.append_basic_block(func, "entry");
        let b = context.create_builder();
        b.position_at_end(bb);
        let s = func.get_nth_param(0).unwrap().into_pointer_value();
        let index = func.get_nth_param(1).unwrap().into_int_value();
        let count = func.get_nth_param(2).unwrap().into_int_value();
        let strlen = module.get_function("strlen").unwrap();
        let slen = b
            .build_call(strlen, &[s.into()], "slen")
            .unwrap()
            .try_as_basic_value()
            .basic()
            .unwrap()
            .into_int_value();
        // start = index - 1; tail_start = start + count; tail_len = slen - tail_start
        let start = b
            .build_int_sub(index, i64_ty.const_int(1, false), "start")
            .unwrap();
        let tail_start = b.build_int_add(start, count, "ts").unwrap();
        let tail_len = b.build_int_sub(slen, tail_start, "tl").unwrap();
        // new_len = start + tail_len
        let new_len = b.build_int_add(start, tail_len, "nl").unwrap();
        let buf_size = b
            .build_int_add(new_len, i64_ty.const_int(1, false), "bsz")
            .unwrap();
        let malloc = module.get_function("malloc").unwrap();
        let buf = b
            .build_call(malloc, &[buf_size.into()], "buf")
            .unwrap()
            .try_as_basic_value()
            .basic()
            .unwrap()
            .into_pointer_value();
        // copy first part
        let memcpy = module.get_function("memcpy").unwrap();
        b.build_call(memcpy, &[buf.into(), s.into(), start.into()], "")
            .unwrap();
        // copy tail
        let buf_tail = unsafe {
            b.build_in_bounds_gep(context.i8_type(), buf, &[start], "bt")
                .unwrap()
        };
        let s_tail = unsafe {
            b.build_in_bounds_gep(context.i8_type(), s, &[tail_start], "st")
                .unwrap()
        };
        let tail_plus1 = b
            .build_int_add(tail_len, i64_ty.const_int(1, false), "tp1")
            .unwrap(); // include null
        b.build_call(
            memcpy,
            &[buf_tail.into(), s_tail.into(), tail_plus1.into()],
            "",
        )
        .unwrap();
        b.build_return(Some(&buf)).unwrap();
    }

    // bruto_str_insert(source: ptr, s: ptr, index: i64) -> ptr
    {
        let fn_ty = ptr_ty.fn_type(&[ptr_ty.into(), ptr_ty.into(), i64_ty.into()], false);
        let func = module.add_function("bruto_str_insert", fn_ty, None);
        let bb = context.append_basic_block(func, "entry");
        let b = context.create_builder();
        b.position_at_end(bb);
        let source = func.get_nth_param(0).unwrap().into_pointer_value();
        let s = func.get_nth_param(1).unwrap().into_pointer_value();
        let index = func.get_nth_param(2).unwrap().into_int_value();
        let strlen = module.get_function("strlen").unwrap();
        let slen = b
            .build_call(strlen, &[s.into()], "slen")
            .unwrap()
            .try_as_basic_value()
            .basic()
            .unwrap()
            .into_int_value();
        let src_len = b
            .build_call(strlen, &[source.into()], "srclen")
            .unwrap()
            .try_as_basic_value()
            .basic()
            .unwrap()
            .into_int_value();
        let total = b.build_int_add(slen, src_len, "tot").unwrap();
        let buf_size = b
            .build_int_add(total, i64_ty.const_int(1, false), "bsz")
            .unwrap();
        let malloc = module.get_function("malloc").unwrap();
        let buf = b
            .build_call(malloc, &[buf_size.into()], "buf")
            .unwrap()
            .try_as_basic_value()
            .basic()
            .unwrap()
            .into_pointer_value();
        let pos = b
            .build_int_sub(index, i64_ty.const_int(1, false), "pos")
            .unwrap();
        // copy s[0..pos]
        let memcpy = module.get_function("memcpy").unwrap();
        b.build_call(memcpy, &[buf.into(), s.into(), pos.into()], "")
            .unwrap();
        // copy source
        let buf_mid = unsafe {
            b.build_in_bounds_gep(context.i8_type(), buf, &[pos], "bm")
                .unwrap()
        };
        b.build_call(memcpy, &[buf_mid.into(), source.into(), src_len.into()], "")
            .unwrap();
        // copy s[pos..] including null
        let buf_tail = unsafe {
            let off = b.build_int_add(pos, src_len, "off").unwrap();
            b.build_in_bounds_gep(context.i8_type(), buf, &[off], "btail")
                .unwrap()
        };
        let s_tail = unsafe {
            b.build_in_bounds_gep(context.i8_type(), s, &[pos], "stail")
                .unwrap()
        };
        let tail_len = b.build_int_sub(slen, pos, "tlen").unwrap();
        let tail_plus1 = b
            .build_int_add(tail_len, i64_ty.const_int(1, false), "tp1")
            .unwrap();
        b.build_call(
            memcpy,
            &[buf_tail.into(), s_tail.into(), tail_plus1.into()],
            "",
        )
        .unwrap();
        b.build_return(Some(&buf)).unwrap();
    }

    // bruto_int_to_str(x: i64) -> ptr
    {
        let fn_ty = ptr_ty.fn_type(&[i64_ty.into()], false);
        let func = module.add_function("bruto_int_to_str", fn_ty, None);
        let bb = context.append_basic_block(func, "entry");
        let b = context.create_builder();
        b.position_at_end(bb);
        let x = func.get_first_param().unwrap().into_int_value();
        let malloc = module.get_function("malloc").unwrap();
        let buf = b
            .build_call(malloc, &[i64_ty.const_int(32, false).into()], "buf")
            .unwrap()
            .try_as_basic_value()
            .basic()
            .unwrap()
            .into_pointer_value();
        let snprintf = module.get_function("snprintf").unwrap();
        let fmt = b.build_global_string_ptr("%lld", "itoa_fmt").unwrap();
        b.build_call(
            snprintf,
            &[
                buf.into(),
                i64_ty.const_int(32, false).into(),
                fmt.as_pointer_value().into(),
                x.into(),
            ],
            "",
        )
        .unwrap();
        b.build_return(Some(&buf)).unwrap();
    }

    // bruto_real_to_str(x: f64) -> ptr
    {
        let f64_ty = context.f64_type();
        let fn_ty = ptr_ty.fn_type(&[f64_ty.into()], false);
        let func = module.add_function("bruto_real_to_str", fn_ty, None);
        let bb = context.append_basic_block(func, "entry");
        let b = context.create_builder();
        b.position_at_end(bb);
        let x = func.get_first_param().unwrap().into_float_value();
        let malloc = module.get_function("malloc").unwrap();
        let buf = b
            .build_call(malloc, &[i64_ty.const_int(64, false).into()], "buf")
            .unwrap()
            .try_as_basic_value()
            .basic()
            .unwrap()
            .into_pointer_value();
        let snprintf = module.get_function("snprintf").unwrap();
        let fmt = b.build_global_string_ptr("%.10g", "ftoa_fmt").unwrap();
        b.build_call(
            snprintf,
            &[
                buf.into(),
                i64_ty.const_int(64, false).into(),
                fmt.as_pointer_value().into(),
                x.into(),
            ],
            "",
        )
        .unwrap();
        b.build_return(Some(&buf)).unwrap();
    }

    // bruto_str_to_int(s: ptr) -> i64
    {
        let fn_ty = i64_ty.fn_type(&[ptr_ty.into()], false);
        let func = module.add_function("bruto_str_to_int", fn_ty, None);
        let bb = context.append_basic_block(func, "entry");
        let b = context.create_builder();
        b.position_at_end(bb);
        let s = func.get_first_param().unwrap().into_pointer_value();
        let atoll = module.get_function("atoll").unwrap();
        let r = b
            .build_call(atoll, &[s.into()], "val")
            .unwrap()
            .try_as_basic_value()
            .basic()
            .unwrap();
        b.build_return(Some(&r)).unwrap();
    }

    // ── Formatted writers (for write(x:N), write(r:N:M)) ─────
    let f64_ty = context.f64_type();

    // bruto_write_int_fmt(val: i64, width: i64)
    {
        let fn_ty = void_ty.fn_type(&[i64_ty.into(), i64_ty.into()], false);
        let func = module.add_function("bruto_write_int_fmt", fn_ty, None);
        let bb = context.append_basic_block(func, "entry");
        let b = context.create_builder();
        b.position_at_end(bb);
        let val = func.get_nth_param(0).unwrap().into_int_value();
        let w = func.get_nth_param(1).unwrap().into_int_value();
        let printf = module.get_function("printf").unwrap();
        let fmt = b.build_global_string_ptr("%*lld", "fmt_int_w").unwrap();
        b.build_call(
            printf,
            &[fmt.as_pointer_value().into(), w.into(), val.into()],
            "",
        )
        .unwrap();
        b.build_return(None).unwrap();
    }
    // capture variant
    {
        let fn_ty = void_ty.fn_type(&[i64_ty.into(), i64_ty.into()], false);
        let func = module.add_function("bruto_capture_write_int_fmt", fn_ty, None);
        let bb = context.append_basic_block(func, "entry");
        let b = context.create_builder();
        b.position_at_end(bb);
        let g = module.get_global("_bruto_capture_fp").unwrap();
        let fp = b
            .build_load(ptr_ty, g.as_pointer_value(), "fp")
            .unwrap()
            .into_pointer_value();
        let is_null = b.build_is_null(fp, "null").unwrap();
        let write_bb = context.append_basic_block(func, "write");
        let end_bb = context.append_basic_block(func, "end");
        b.build_conditional_branch(is_null, end_bb, write_bb)
            .unwrap();
        b.position_at_end(write_bb);
        let val = func.get_nth_param(0).unwrap().into_int_value();
        let w = func.get_nth_param(1).unwrap().into_int_value();
        let fprintf = module.get_function("fprintf").unwrap();
        let fmt = b.build_global_string_ptr("%*lld", "cap_fmt_int_w").unwrap();
        b.build_call(
            fprintf,
            &[
                fp.into(),
                fmt.as_pointer_value().into(),
                w.into(),
                val.into(),
            ],
            "",
        )
        .unwrap();
        let fflush = module.get_function("fflush").unwrap();
        b.build_call(fflush, &[fp.into()], "").unwrap();
        b.build_unconditional_branch(end_bb).unwrap();
        b.position_at_end(end_bb);
        b.build_return(None).unwrap();
    }

    // bruto_write_real_fmt(val: f64, width: i64, prec: i64)
    {
        let fn_ty = void_ty.fn_type(&[f64_ty.into(), i64_ty.into(), i64_ty.into()], false);
        let func = module.add_function("bruto_write_real_fmt", fn_ty, None);
        let bb = context.append_basic_block(func, "entry");
        let b = context.create_builder();
        b.position_at_end(bb);
        let val = func.get_nth_param(0).unwrap().into_float_value();
        let w = func.get_nth_param(1).unwrap().into_int_value();
        let p = func.get_nth_param(2).unwrap().into_int_value();
        let printf = module.get_function("printf").unwrap();
        let fmt = b.build_global_string_ptr("%*.*f", "fmt_real_wp").unwrap();
        b.build_call(
            printf,
            &[
                fmt.as_pointer_value().into(),
                w.into(),
                p.into(),
                val.into(),
            ],
            "",
        )
        .unwrap();
        b.build_return(None).unwrap();
    }
    {
        let fn_ty = void_ty.fn_type(&[f64_ty.into(), i64_ty.into(), i64_ty.into()], false);
        let func = module.add_function("bruto_capture_write_real_fmt", fn_ty, None);
        let bb = context.append_basic_block(func, "entry");
        let b = context.create_builder();
        b.position_at_end(bb);
        let g = module.get_global("_bruto_capture_fp").unwrap();
        let fp = b
            .build_load(ptr_ty, g.as_pointer_value(), "fp")
            .unwrap()
            .into_pointer_value();
        let is_null = b.build_is_null(fp, "null").unwrap();
        let write_bb = context.append_basic_block(func, "write");
        let end_bb = context.append_basic_block(func, "end");
        b.build_conditional_branch(is_null, end_bb, write_bb)
            .unwrap();
        b.position_at_end(write_bb);
        let val = func.get_nth_param(0).unwrap().into_float_value();
        let w = func.get_nth_param(1).unwrap().into_int_value();
        let p = func.get_nth_param(2).unwrap().into_int_value();
        let fprintf = module.get_function("fprintf").unwrap();
        let fmt = b
            .build_global_string_ptr("%*.*f", "cap_fmt_real_wp")
            .unwrap();
        b.build_call(
            fprintf,
            &[
                fp.into(),
                fmt.as_pointer_value().into(),
                w.into(),
                p.into(),
                val.into(),
            ],
            "",
        )
        .unwrap();
        let fflush = module.get_function("fflush").unwrap();
        b.build_call(fflush, &[fp.into()], "").unwrap();
        b.build_unconditional_branch(end_bb).unwrap();
        b.position_at_end(end_bb);
        b.build_return(None).unwrap();
    }

    // bruto_write_real_fmt_w(val: f64, width: i64) — width-only, default precision
    {
        let fn_ty = void_ty.fn_type(&[f64_ty.into(), i64_ty.into()], false);
        let func = module.add_function("bruto_write_real_fmt_w", fn_ty, None);
        let bb = context.append_basic_block(func, "entry");
        let b = context.create_builder();
        b.position_at_end(bb);
        let val = func.get_nth_param(0).unwrap().into_float_value();
        let w = func.get_nth_param(1).unwrap().into_int_value();
        let printf = module.get_function("printf").unwrap();
        let fmt = b.build_global_string_ptr("%*.10g", "fmt_real_w").unwrap();
        b.build_call(
            printf,
            &[fmt.as_pointer_value().into(), w.into(), val.into()],
            "",
        )
        .unwrap();
        b.build_return(None).unwrap();
    }
    {
        let fn_ty = void_ty.fn_type(&[f64_ty.into(), i64_ty.into()], false);
        let func = module.add_function("bruto_capture_write_real_fmt_w", fn_ty, None);
        let bb = context.append_basic_block(func, "entry");
        let b = context.create_builder();
        b.position_at_end(bb);
        let g = module.get_global("_bruto_capture_fp").unwrap();
        let fp = b
            .build_load(ptr_ty, g.as_pointer_value(), "fp")
            .unwrap()
            .into_pointer_value();
        let is_null = b.build_is_null(fp, "null").unwrap();
        let write_bb = context.append_basic_block(func, "write");
        let end_bb = context.append_basic_block(func, "end");
        b.build_conditional_branch(is_null, end_bb, write_bb)
            .unwrap();
        b.position_at_end(write_bb);
        let val = func.get_nth_param(0).unwrap().into_float_value();
        let w = func.get_nth_param(1).unwrap().into_int_value();
        let fprintf = module.get_function("fprintf").unwrap();
        let fmt = b
            .build_global_string_ptr("%*.10g", "cap_fmt_real_w")
            .unwrap();
        b.build_call(
            fprintf,
            &[
                fp.into(),
                fmt.as_pointer_value().into(),
                w.into(),
                val.into(),
            ],
            "",
        )
        .unwrap();
        let fflush = module.get_function("fflush").unwrap();
        b.build_call(fflush, &[fp.into()], "").unwrap();
        b.build_unconditional_branch(end_bb).unwrap();
        b.position_at_end(end_bb);
        b.build_return(None).unwrap();
    }

    // bruto_write_str_fmt(s: ptr, width: i64)
    {
        let fn_ty = void_ty.fn_type(&[ptr_ty.into(), i64_ty.into()], false);
        let func = module.add_function("bruto_write_str_fmt", fn_ty, None);
        let bb = context.append_basic_block(func, "entry");
        let b = context.create_builder();
        b.position_at_end(bb);
        let s = func.get_nth_param(0).unwrap().into_pointer_value();
        let w = func.get_nth_param(1).unwrap().into_int_value();
        let printf = module.get_function("printf").unwrap();
        let fmt = b.build_global_string_ptr("%*s", "fmt_str_w").unwrap();
        b.build_call(
            printf,
            &[fmt.as_pointer_value().into(), w.into(), s.into()],
            "",
        )
        .unwrap();
        b.build_return(None).unwrap();
    }
    {
        let fn_ty = void_ty.fn_type(&[ptr_ty.into(), i64_ty.into()], false);
        let func = module.add_function("bruto_capture_write_str_fmt", fn_ty, None);
        let bb = context.append_basic_block(func, "entry");
        let b = context.create_builder();
        b.position_at_end(bb);
        let g = module.get_global("_bruto_capture_fp").unwrap();
        let fp = b
            .build_load(ptr_ty, g.as_pointer_value(), "fp")
            .unwrap()
            .into_pointer_value();
        let is_null = b.build_is_null(fp, "null").unwrap();
        let write_bb = context.append_basic_block(func, "write");
        let end_bb = context.append_basic_block(func, "end");
        b.build_conditional_branch(is_null, end_bb, write_bb)
            .unwrap();
        b.position_at_end(write_bb);
        let s = func.get_nth_param(0).unwrap().into_pointer_value();
        let w = func.get_nth_param(1).unwrap().into_int_value();
        let fprintf = module.get_function("fprintf").unwrap();
        let fmt = b.build_global_string_ptr("%*s", "cap_fmt_str_w").unwrap();
        b.build_call(
            fprintf,
            &[fp.into(), fmt.as_pointer_value().into(), w.into(), s.into()],
            "",
        )
        .unwrap();
        let fflush = module.get_function("fflush").unwrap();
        b.build_call(fflush, &[fp.into()], "").unwrap();
        b.build_unconditional_branch(end_bb).unwrap();
        b.position_at_end(end_bb);
        b.build_return(None).unwrap();
    }

    // ── Extra readers for readln(real / char / string) ─────
    {
        let fn_ty = f64_ty.fn_type(&[], false);
        let func = module.add_function("bruto_read_real", fn_ty, None);
        let bb = context.append_basic_block(func, "entry");
        let b = context.create_builder();
        b.position_at_end(bb);
        let tmp = b.build_alloca(f64_ty, "tmp").unwrap();
        b.build_store(tmp, f64_ty.const_float(0.0)).unwrap();
        let scanf = module.get_function("scanf").unwrap();
        let fmt = b.build_global_string_ptr("%lf", "fmt_lf").unwrap();
        b.build_call(scanf, &[fmt.as_pointer_value().into(), tmp.into()], "")
            .unwrap();
        let val = b.build_load(f64_ty, tmp, "v").unwrap();
        b.build_return(Some(&val)).unwrap();
    }
    {
        let i8_ty = context.i8_type();
        let fn_ty = i8_ty.fn_type(&[], false);
        let func = module.add_function("bruto_read_char", fn_ty, None);
        let bb = context.append_basic_block(func, "entry");
        let b = context.create_builder();
        b.position_at_end(bb);
        let getchar = module.get_function("getchar").unwrap_or_else(|| {
            let ft = i32_ty.fn_type(&[], false);
            module.add_function("getchar", ft, None)
        });
        let r = b
            .build_call(getchar, &[], "c")
            .unwrap()
            .try_as_basic_value()
            .basic()
            .unwrap()
            .into_int_value();
        let trunc = b.build_int_truncate(r, i8_ty, "ct").unwrap();
        b.build_return(Some(&trunc)).unwrap();
    }
    {
        // bruto_read_str() -> ptr  (reads up to newline into a fresh heap buffer)
        let fn_ty = ptr_ty.fn_type(&[], false);
        let func = module.add_function("bruto_read_str", fn_ty, None);
        let bb = context.append_basic_block(func, "entry");
        let b = context.create_builder();
        b.position_at_end(bb);
        let malloc = module.get_function("malloc").unwrap();
        let buf_size = i64_ty.const_int(1024, false);
        let buf = b
            .build_call(malloc, &[buf_size.into()], "buf")
            .unwrap()
            .try_as_basic_value()
            .basic()
            .unwrap()
            .into_pointer_value();
        b.build_store(buf, context.i8_type().const_int(0, false))
            .unwrap();
        let fgets_fn = module.get_function("fgets").unwrap_or_else(|| {
            let ft = ptr_ty.fn_type(&[ptr_ty.into(), i32_ty.into(), ptr_ty.into()], false);
            module.add_function("fgets", ft, None)
        });
        let stdin_fp =
            crate::target::emit_load_stdio(&b, module, context, crate::target::Stdio::Stdin);
        b.build_call(
            fgets_fn,
            &[
                buf.into(),
                i32_ty.const_int(1024, false).into(),
                stdin_fp.into(),
            ],
            "",
        )
        .unwrap();
        let strlen = module.get_function("strlen").unwrap();
        let len = b
            .build_call(strlen, &[buf.into()], "L")
            .unwrap()
            .try_as_basic_value()
            .basic()
            .unwrap()
            .into_int_value();
        let zero_check = b
            .build_int_compare(
                inkwell::IntPredicate::EQ,
                len,
                i64_ty.const_int(0, false),
                "is0",
            )
            .unwrap();
        let not_zero_bb = context.append_basic_block(func, "not_zero");
        let done_bb = context.append_basic_block(func, "done");
        b.build_conditional_branch(zero_check, done_bb, not_zero_bb)
            .unwrap();
        b.position_at_end(not_zero_bb);
        let last_idx = b
            .build_int_sub(len, i64_ty.const_int(1, false), "li")
            .unwrap();
        let last_ptr = unsafe {
            b.build_in_bounds_gep(context.i8_type(), buf, &[last_idx], "lp")
                .unwrap()
        };
        let last_ch = b
            .build_load(context.i8_type(), last_ptr, "lc")
            .unwrap()
            .into_int_value();
        let is_nl = b
            .build_int_compare(
                inkwell::IntPredicate::EQ,
                last_ch,
                context.i8_type().const_int(10, false),
                "isnl",
            )
            .unwrap();
        let strip_bb = context.append_basic_block(func, "strip");
        b.build_conditional_branch(is_nl, strip_bb, done_bb)
            .unwrap();
        b.position_at_end(strip_bb);
        b.build_store(last_ptr, context.i8_type().const_int(0, false))
            .unwrap();
        b.build_unconditional_branch(done_bb).unwrap();
        b.position_at_end(done_bb);
        b.build_return(Some(&buf)).unwrap();
    }

    // ── File I/O runtime (text files) ───────────────────────
    // bruto_file is opaque to LLVM — we hand it as i8*.
    // Layout: { FILE* fp; char* path; }

    // bruto_file_new(path: ptr) -> ptr
    // Layout: { FILE* fp; char* path; i8 buf; i8 valid; [6 bytes pad] }  = 24 bytes
    {
        let fn_ty = ptr_ty.fn_type(&[ptr_ty.into()], false);
        let func = module.add_function("bruto_file_new", fn_ty, None);
        let bb = context.append_basic_block(func, "entry");
        let b = context.create_builder();
        b.position_at_end(bb);
        let path = func.get_first_param().unwrap().into_pointer_value();
        let malloc = module.get_function("malloc").unwrap();
        let s = b
            .build_call(malloc, &[i64_ty.const_int(24, false).into()], "s")
            .unwrap()
            .try_as_basic_value()
            .basic()
            .unwrap()
            .into_pointer_value();
        b.build_store(s, ptr_ty.const_null()).unwrap();
        let strdup_fn = module.get_function("strdup").unwrap_or_else(|| {
            let ft = ptr_ty.fn_type(&[ptr_ty.into()], false);
            module.add_function("strdup", ft, None)
        });
        let dup = b
            .build_call(strdup_fn, &[path.into()], "d")
            .unwrap()
            .try_as_basic_value()
            .basic()
            .unwrap();
        let path_ptr = unsafe {
            b.build_in_bounds_gep(context.i8_type(), s, &[i64_ty.const_int(8, false)], "pp")
                .unwrap()
        };
        b.build_store(path_ptr, dup).unwrap();
        // Initialize buffer byte and valid flag to 0
        let buf_ptr = unsafe {
            b.build_in_bounds_gep(
                context.i8_type(),
                s,
                &[i64_ty.const_int(16, false)],
                "buf_init",
            )
            .unwrap()
        };
        b.build_store(buf_ptr, context.i8_type().const_int(0, false))
            .unwrap();
        let valid_ptr = unsafe {
            b.build_in_bounds_gep(
                context.i8_type(),
                s,
                &[i64_ty.const_int(17, false)],
                "v_init",
            )
            .unwrap()
        };
        b.build_store(valid_ptr, context.i8_type().const_int(0, false))
            .unwrap();
        b.build_return(Some(&s)).unwrap();
    }

    // bruto_file_open(f: ptr, mode: ptr) — fopens with given mode
    {
        let fn_ty = void_ty.fn_type(&[ptr_ty.into(), ptr_ty.into()], false);
        let func = module.add_function("bruto_file_open", fn_ty, None);
        let bb = context.append_basic_block(func, "entry");
        let b = context.create_builder();
        b.position_at_end(bb);
        let s = func.get_nth_param(0).unwrap().into_pointer_value();
        let mode = func.get_nth_param(1).unwrap().into_pointer_value();
        let path_ptr = unsafe {
            b.build_in_bounds_gep(context.i8_type(), s, &[i64_ty.const_int(8, false)], "pp")
                .unwrap()
        };
        let path = b
            .build_load(ptr_ty, path_ptr, "path")
            .unwrap()
            .into_pointer_value();
        let fopen = module.get_function("fopen").unwrap();
        let fp = b
            .build_call(fopen, &[path.into(), mode.into()], "fp")
            .unwrap()
            .try_as_basic_value()
            .basic()
            .unwrap();
        b.build_store(s, fp).unwrap();
        b.build_return(None).unwrap();
    }

    // bruto_file_close(f: ptr)
    {
        let fn_ty = void_ty.fn_type(&[ptr_ty.into()], false);
        let func = module.add_function("bruto_file_close", fn_ty, None);
        let bb = context.append_basic_block(func, "entry");
        let b = context.create_builder();
        b.position_at_end(bb);
        let s = func.get_first_param().unwrap().into_pointer_value();
        let fp = b.build_load(ptr_ty, s, "fp").unwrap().into_pointer_value();
        let is_null = b.build_is_null(fp, "null").unwrap();
        let close_bb = context.append_basic_block(func, "close");
        let end_bb = context.append_basic_block(func, "end");
        b.build_conditional_branch(is_null, end_bb, close_bb)
            .unwrap();
        b.position_at_end(close_bb);
        let fclose = module.get_function("fclose").unwrap();
        b.build_call(fclose, &[fp.into()], "").unwrap();
        b.build_store(s, ptr_ty.const_null()).unwrap();
        b.build_unconditional_branch(end_bb).unwrap();
        b.position_at_end(end_bb);
        b.build_return(None).unwrap();
    }

    // bruto_file_eof(f: ptr) -> i1
    {
        let fn_ty = i1_ty.fn_type(&[ptr_ty.into()], false);
        let func = module.add_function("bruto_file_eof", fn_ty, None);
        let bb = context.append_basic_block(func, "entry");
        let b = context.create_builder();
        b.position_at_end(bb);
        let s = func.get_first_param().unwrap().into_pointer_value();
        let fp = b.build_load(ptr_ty, s, "fp").unwrap().into_pointer_value();
        let is_null = b.build_is_null(fp, "null").unwrap();
        let chk_bb = context.append_basic_block(func, "chk");
        let null_bb = context.append_basic_block(func, "null_done");
        let end_bb = context.append_basic_block(func, "end");
        b.build_conditional_branch(is_null, null_bb, chk_bb)
            .unwrap();
        b.position_at_end(null_bb);
        b.build_unconditional_branch(end_bb).unwrap();
        b.position_at_end(chk_bb);
        let feof = module.get_function("feof").unwrap_or_else(|| {
            let ft = i32_ty.fn_type(&[ptr_ty.into()], false);
            module.add_function("feof", ft, None)
        });
        let r = b
            .build_call(feof, &[fp.into()], "r")
            .unwrap()
            .try_as_basic_value()
            .basic()
            .unwrap()
            .into_int_value();
        let nz = b
            .build_int_compare(
                inkwell::IntPredicate::NE,
                r,
                i32_ty.const_int(0, false),
                "nz",
            )
            .unwrap();
        b.build_unconditional_branch(end_bb).unwrap();
        b.position_at_end(end_bb);
        let phi = b.build_phi(i1_ty, "eof_phi").unwrap();
        phi.add_incoming(&[(&i1_ty.const_int(1, false), null_bb), (&nz, chk_bb)]);
        b.build_return(Some(&phi.as_basic_value())).unwrap();
    }

    // bruto_file_write_str(f: ptr, s: ptr)
    {
        let fn_ty = void_ty.fn_type(&[ptr_ty.into(), ptr_ty.into()], false);
        let func = module.add_function("bruto_file_write_str", fn_ty, None);
        let bb = context.append_basic_block(func, "entry");
        let b = context.create_builder();
        b.position_at_end(bb);
        let s = func.get_nth_param(0).unwrap().into_pointer_value();
        let txt = func.get_nth_param(1).unwrap().into_pointer_value();
        let fp = b.build_load(ptr_ty, s, "fp").unwrap().into_pointer_value();
        let fprintf = module.get_function("fprintf").unwrap();
        let fmt = b.build_global_string_ptr("%s", "fwfmt_s").unwrap();
        b.build_call(
            fprintf,
            &[fp.into(), fmt.as_pointer_value().into(), txt.into()],
            "",
        )
        .unwrap();
        b.build_return(None).unwrap();
    }

    // bruto_file_write_int(f: ptr, val: i64)
    {
        let fn_ty = void_ty.fn_type(&[ptr_ty.into(), i64_ty.into()], false);
        let func = module.add_function("bruto_file_write_int", fn_ty, None);
        let bb = context.append_basic_block(func, "entry");
        let b = context.create_builder();
        b.position_at_end(bb);
        let s = func.get_nth_param(0).unwrap().into_pointer_value();
        let v = func.get_nth_param(1).unwrap().into_int_value();
        let fp = b.build_load(ptr_ty, s, "fp").unwrap().into_pointer_value();
        let fprintf = module.get_function("fprintf").unwrap();
        let fmt = b.build_global_string_ptr("%lld", "fwfmt_i").unwrap();
        b.build_call(
            fprintf,
            &[fp.into(), fmt.as_pointer_value().into(), v.into()],
            "",
        )
        .unwrap();
        b.build_return(None).unwrap();
    }

    // bruto_file_write_real(f: ptr, val: f64)
    {
        let fn_ty = void_ty.fn_type(&[ptr_ty.into(), f64_ty.into()], false);
        let func = module.add_function("bruto_file_write_real", fn_ty, None);
        let bb = context.append_basic_block(func, "entry");
        let b = context.create_builder();
        b.position_at_end(bb);
        let s = func.get_nth_param(0).unwrap().into_pointer_value();
        let v = func.get_nth_param(1).unwrap().into_float_value();
        let fp = b.build_load(ptr_ty, s, "fp").unwrap().into_pointer_value();
        let fprintf = module.get_function("fprintf").unwrap();
        let fmt = b.build_global_string_ptr("%.10g", "fwfmt_f").unwrap();
        b.build_call(
            fprintf,
            &[fp.into(), fmt.as_pointer_value().into(), v.into()],
            "",
        )
        .unwrap();
        b.build_return(None).unwrap();
    }

    // bruto_file_write_char(f: ptr, val: i8)
    {
        let i8_ty = context.i8_type();
        let fn_ty = void_ty.fn_type(&[ptr_ty.into(), i8_ty.into()], false);
        let func = module.add_function("bruto_file_write_char", fn_ty, None);
        let bb = context.append_basic_block(func, "entry");
        let b = context.create_builder();
        b.position_at_end(bb);
        let s = func.get_nth_param(0).unwrap().into_pointer_value();
        let v = func.get_nth_param(1).unwrap().into_int_value();
        let fp = b.build_load(ptr_ty, s, "fp").unwrap().into_pointer_value();
        let fputc_fn = module.get_function("fputc").unwrap_or_else(|| {
            let ft = i32_ty.fn_type(&[i32_ty.into(), ptr_ty.into()], false);
            module.add_function("fputc", ft, None)
        });
        let ext = b.build_int_z_extend(v, i32_ty, "z").unwrap();
        b.build_call(fputc_fn, &[ext.into(), fp.into()], "")
            .unwrap();
        b.build_return(None).unwrap();
    }

    // bruto_file_writeln(f: ptr)
    {
        let fn_ty = void_ty.fn_type(&[ptr_ty.into()], false);
        let func = module.add_function("bruto_file_writeln", fn_ty, None);
        let bb = context.append_basic_block(func, "entry");
        let b = context.create_builder();
        b.position_at_end(bb);
        let s = func.get_first_param().unwrap().into_pointer_value();
        let fp = b.build_load(ptr_ty, s, "fp").unwrap().into_pointer_value();
        let fputc_fn = module.get_function("fputc").unwrap_or_else(|| {
            let ft = i32_ty.fn_type(&[i32_ty.into(), ptr_ty.into()], false);
            module.add_function("fputc", ft, None)
        });
        b.build_call(
            fputc_fn,
            &[i32_ty.const_int(10, false).into(), fp.into()],
            "",
        )
        .unwrap();
        b.build_return(None).unwrap();
    }

    // bruto_file_read_int(f: ptr) -> i64
    {
        let fn_ty = i64_ty.fn_type(&[ptr_ty.into()], false);
        let func = module.add_function("bruto_file_read_int", fn_ty, None);
        let bb = context.append_basic_block(func, "entry");
        let b = context.create_builder();
        b.position_at_end(bb);
        let s = func.get_first_param().unwrap().into_pointer_value();
        let fp = b.build_load(ptr_ty, s, "fp").unwrap().into_pointer_value();
        let tmp = b.build_alloca(i64_ty, "tmp").unwrap();
        b.build_store(tmp, i64_ty.const_int(0, false)).unwrap();
        let fscanf = module.get_function("fscanf").unwrap_or_else(|| {
            let ft = i32_ty.fn_type(&[ptr_ty.into(), ptr_ty.into()], true);
            module.add_function("fscanf", ft, None)
        });
        let fmt = b.build_global_string_ptr("%lld", "frfmt_i").unwrap();
        b.build_call(
            fscanf,
            &[fp.into(), fmt.as_pointer_value().into(), tmp.into()],
            "",
        )
        .unwrap();
        let val = b.build_load(i64_ty, tmp, "v").unwrap();
        b.build_return(Some(&val)).unwrap();
    }

    // bruto_file_read_real(f: ptr) -> f64
    {
        let fn_ty = f64_ty.fn_type(&[ptr_ty.into()], false);
        let func = module.add_function("bruto_file_read_real", fn_ty, None);
        let bb = context.append_basic_block(func, "entry");
        let b = context.create_builder();
        b.position_at_end(bb);
        let s = func.get_first_param().unwrap().into_pointer_value();
        let fp = b.build_load(ptr_ty, s, "fp").unwrap().into_pointer_value();
        let tmp = b.build_alloca(f64_ty, "tmp").unwrap();
        b.build_store(tmp, f64_ty.const_float(0.0)).unwrap();
        let fscanf = module.get_function("fscanf").unwrap_or_else(|| {
            let ft = i32_ty.fn_type(&[ptr_ty.into(), ptr_ty.into()], true);
            module.add_function("fscanf", ft, None)
        });
        let fmt = b.build_global_string_ptr("%lf", "frfmt_f").unwrap();
        b.build_call(
            fscanf,
            &[fp.into(), fmt.as_pointer_value().into(), tmp.into()],
            "",
        )
        .unwrap();
        let val = b.build_load(f64_ty, tmp, "v").unwrap();
        b.build_return(Some(&val)).unwrap();
    }

    // bruto_file_read_char(f: ptr) -> i8
    {
        let i8_ty = context.i8_type();
        let fn_ty = i8_ty.fn_type(&[ptr_ty.into()], false);
        let func = module.add_function("bruto_file_read_char", fn_ty, None);
        let bb = context.append_basic_block(func, "entry");
        let b = context.create_builder();
        b.position_at_end(bb);
        let s = func.get_first_param().unwrap().into_pointer_value();
        let fp = b.build_load(ptr_ty, s, "fp").unwrap().into_pointer_value();
        let fgetc_fn = module.get_function("fgetc").unwrap_or_else(|| {
            let ft = i32_ty.fn_type(&[ptr_ty.into()], false);
            module.add_function("fgetc", ft, None)
        });
        let r = b
            .build_call(fgetc_fn, &[fp.into()], "r")
            .unwrap()
            .try_as_basic_value()
            .basic()
            .unwrap()
            .into_int_value();
        let trunc = b.build_int_truncate(r, i8_ty, "ct").unwrap();
        b.build_return(Some(&trunc)).unwrap();
    }

    // ── Stack overflow / signal handling ────────────────────
    // bruto_segv_handler(sig: i32) — writes a message and exits.
    // bruto_install_stack_guard() — calls signal(SIGSEGV)/signal(SIGBUS).
    {
        // Declare libc bits we'll need.
        let write_fn = module.get_function("write").unwrap_or_else(|| {
            let ft = i64_ty.fn_type(&[i32_ty.into(), ptr_ty.into(), i64_ty.into()], false);
            module.add_function("write", ft, None)
        });
        let exit_fn = module.get_function("_exit").unwrap_or_else(|| {
            let ft = void_ty.fn_type(&[i32_ty.into()], false);
            module.add_function("_exit", ft, None)
        });
        // signal(int, void(*)(int)) -> void(*)(int)
        let signal_fn = module.get_function("signal").unwrap_or_else(|| {
            let ft = ptr_ty.fn_type(&[i32_ty.into(), ptr_ty.into()], false);
            module.add_function("signal", ft, None)
        });

        // bruto_segv_handler(sig: i32)
        let handler_ty = void_ty.fn_type(&[i32_ty.into()], false);
        let handler = module.add_function("bruto_segv_handler", handler_ty, None);
        let bb = context.append_basic_block(handler, "entry");
        let b = context.create_builder();
        b.position_at_end(bb);
        let msg = "Runtime error: stack overflow or memory access violation\n";
        let s = b.build_global_string_ptr(msg, "segv_msg").unwrap();
        b.build_call(
            write_fn,
            &[
                i32_ty.const_int(2, false).into(),
                s.as_pointer_value().into(),
                i64_ty.const_int(msg.len() as u64, false).into(),
            ],
            "",
        )
        .unwrap();
        b.build_call(exit_fn, &[i32_ty.const_int(1, false).into()], "")
            .unwrap();
        b.build_unreachable().unwrap();

        // bruto_install_stack_guard()
        let install_ty = void_ty.fn_type(&[], false);
        let install = module.add_function("bruto_install_stack_guard", install_ty, None);
        let bb = context.append_basic_block(install, "entry");
        let b = context.create_builder();
        b.position_at_end(bb);
        let h_ptr = handler.as_global_value().as_pointer_value();
        // SIGSEGV = 11
        b.build_call(
            signal_fn,
            &[i32_ty.const_int(11, false).into(), h_ptr.into()],
            "",
        )
        .unwrap();
        // SIGBUS = 10 on macOS, 7 on Linux. Install for both to be safe.
        b.build_call(
            signal_fn,
            &[i32_ty.const_int(10, false).into(), h_ptr.into()],
            "",
        )
        .unwrap();
        b.build_call(
            signal_fn,
            &[i32_ty.const_int(7, false).into(), h_ptr.into()],
            "",
        )
        .unwrap();
        b.build_return(None).unwrap();
    }

    // bruto_make_predef_file(fp: ptr) -> ptr  (allocates a bruto_file with given FILE*)
    {
        let fn_ty = ptr_ty.fn_type(&[ptr_ty.into()], false);
        let func = module.add_function("bruto_make_predef_file", fn_ty, None);
        let bb = context.append_basic_block(func, "entry");
        let b = context.create_builder();
        b.position_at_end(bb);
        let fp = func.get_first_param().unwrap().into_pointer_value();
        let malloc = module.get_function("malloc").unwrap();
        let s = b
            .build_call(malloc, &[i64_ty.const_int(24, false).into()], "s")
            .unwrap()
            .try_as_basic_value()
            .basic()
            .unwrap()
            .into_pointer_value();
        b.build_store(s, fp).unwrap();
        let path_ptr = unsafe {
            b.build_in_bounds_gep(context.i8_type(), s, &[i64_ty.const_int(8, false)], "pp")
                .unwrap()
        };
        b.build_store(path_ptr, ptr_ty.const_null()).unwrap();
        let buf_ptr = unsafe {
            b.build_in_bounds_gep(
                context.i8_type(),
                s,
                &[i64_ty.const_int(16, false)],
                "buf_init",
            )
            .unwrap()
        };
        b.build_store(buf_ptr, context.i8_type().const_int(0, false))
            .unwrap();
        let valid_ptr = unsafe {
            b.build_in_bounds_gep(
                context.i8_type(),
                s,
                &[i64_ty.const_int(17, false)],
                "v_init",
            )
            .unwrap()
        };
        b.build_store(valid_ptr, context.i8_type().const_int(0, false))
            .unwrap();
        b.build_return(Some(&s)).unwrap();
    }

    // bruto_file_get(f: ptr) — advance buffer: f^ := next char
    {
        let fn_ty = void_ty.fn_type(&[ptr_ty.into()], false);
        let func = module.add_function("bruto_file_get", fn_ty, None);
        let bb = context.append_basic_block(func, "entry");
        let b = context.create_builder();
        b.position_at_end(bb);
        let s = func.get_first_param().unwrap().into_pointer_value();
        let fp = b.build_load(ptr_ty, s, "fp").unwrap().into_pointer_value();
        let fgetc_fn = module.get_function("fgetc").unwrap_or_else(|| {
            let ft = i32_ty.fn_type(&[ptr_ty.into()], false);
            module.add_function("fgetc", ft, None)
        });
        let c = b
            .build_call(fgetc_fn, &[fp.into()], "c")
            .unwrap()
            .try_as_basic_value()
            .basic()
            .unwrap()
            .into_int_value();
        let trunc = b.build_int_truncate(c, context.i8_type(), "ct").unwrap();
        let buf_ptr = unsafe {
            b.build_in_bounds_gep(context.i8_type(), s, &[i64_ty.const_int(16, false)], "bp")
                .unwrap()
        };
        b.build_store(buf_ptr, trunc).unwrap();
        // valid := (c != -1)
        let is_eof = b
            .build_int_compare(
                inkwell::IntPredicate::EQ,
                c,
                i32_ty.const_int((-1i32) as u64, true),
                "is_eof",
            )
            .unwrap();
        let zero = context.i8_type().const_int(0, false);
        let one = context.i8_type().const_int(1, false);
        let valid = b.build_select(is_eof, zero, one, "valid").unwrap();
        let valid_ptr = unsafe {
            b.build_in_bounds_gep(context.i8_type(), s, &[i64_ty.const_int(17, false)], "vp")
                .unwrap()
        };
        b.build_store(valid_ptr, valid).unwrap();
        b.build_return(None).unwrap();
    }

    // bruto_file_put(f: ptr) — write buffer to file
    {
        let fn_ty = void_ty.fn_type(&[ptr_ty.into()], false);
        let func = module.add_function("bruto_file_put", fn_ty, None);
        let bb = context.append_basic_block(func, "entry");
        let b = context.create_builder();
        b.position_at_end(bb);
        let s = func.get_first_param().unwrap().into_pointer_value();
        let fp = b.build_load(ptr_ty, s, "fp").unwrap().into_pointer_value();
        let buf_ptr = unsafe {
            b.build_in_bounds_gep(context.i8_type(), s, &[i64_ty.const_int(16, false)], "bp")
                .unwrap()
        };
        let buf = b
            .build_load(context.i8_type(), buf_ptr, "buf")
            .unwrap()
            .into_int_value();
        let fputc_fn = module.get_function("fputc").unwrap_or_else(|| {
            let ft = i32_ty.fn_type(&[i32_ty.into(), ptr_ty.into()], false);
            module.add_function("fputc", ft, None)
        });
        let ext = b.build_int_z_extend(buf, i32_ty, "z").unwrap();
        b.build_call(fputc_fn, &[ext.into(), fp.into()], "")
            .unwrap();
        b.build_return(None).unwrap();
    }

    // bruto_file_buf_load(f: ptr) -> i8
    {
        let i8_ty = context.i8_type();
        let fn_ty = i8_ty.fn_type(&[ptr_ty.into()], false);
        let func = module.add_function("bruto_file_buf_load", fn_ty, None);
        let bb = context.append_basic_block(func, "entry");
        let b = context.create_builder();
        b.position_at_end(bb);
        let s = func.get_first_param().unwrap().into_pointer_value();
        let buf_ptr = unsafe {
            b.build_in_bounds_gep(i8_ty, s, &[i64_ty.const_int(16, false)], "bp")
                .unwrap()
        };
        let v = b.build_load(i8_ty, buf_ptr, "v").unwrap();
        b.build_return(Some(&v)).unwrap();
    }

    // bruto_file_buf_store(f: ptr, c: i8)
    {
        let i8_ty = context.i8_type();
        let fn_ty = void_ty.fn_type(&[ptr_ty.into(), i8_ty.into()], false);
        let func = module.add_function("bruto_file_buf_store", fn_ty, None);
        let bb = context.append_basic_block(func, "entry");
        let b = context.create_builder();
        b.position_at_end(bb);
        let s = func.get_nth_param(0).unwrap().into_pointer_value();
        let c = func.get_nth_param(1).unwrap().into_int_value();
        let buf_ptr = unsafe {
            b.build_in_bounds_gep(i8_ty, s, &[i64_ty.const_int(16, false)], "bp")
                .unwrap()
        };
        b.build_store(buf_ptr, c).unwrap();
        b.build_return(None).unwrap();
    }

    // bruto_ioresult() -> i32  (returns and clears the error code)
    {
        let fn_ty = i32_ty.fn_type(&[], false);
        let func = module.add_function("bruto_ioresult", fn_ty, None);
        let bb = context.append_basic_block(func, "entry");
        let b = context.create_builder();
        b.position_at_end(bb);
        let g = module.get_global("_bruto_ioresult").unwrap();
        let v = b.build_load(i32_ty, g.as_pointer_value(), "v").unwrap();
        b.build_store(g.as_pointer_value(), i32_ty.const_int(0, false))
            .unwrap();
        b.build_return(Some(&v)).unwrap();
    }

    // bruto_file_eoln(f: ptr) -> i1  (peeks next char, returns true if \n or EOF)
    {
        let fn_ty = i1_ty.fn_type(&[ptr_ty.into()], false);
        let func = module.add_function("bruto_file_eoln", fn_ty, None);
        let bb = context.append_basic_block(func, "entry");
        let b = context.create_builder();
        b.position_at_end(bb);
        let s = func.get_first_param().unwrap().into_pointer_value();
        let fp = b.build_load(ptr_ty, s, "fp").unwrap().into_pointer_value();
        let is_null = b.build_is_null(fp, "null").unwrap();
        let chk_bb = context.append_basic_block(func, "chk");
        let null_bb = context.append_basic_block(func, "null_done");
        let end_bb = context.append_basic_block(func, "end");
        b.build_conditional_branch(is_null, null_bb, chk_bb)
            .unwrap();
        b.position_at_end(null_bb);
        b.build_unconditional_branch(end_bb).unwrap();
        b.position_at_end(chk_bb);
        let fgetc_fn = module.get_function("fgetc").unwrap_or_else(|| {
            let ft = i32_ty.fn_type(&[ptr_ty.into()], false);
            module.add_function("fgetc", ft, None)
        });
        let ungetc_fn = module.get_function("ungetc").unwrap_or_else(|| {
            let ft = i32_ty.fn_type(&[i32_ty.into(), ptr_ty.into()], false);
            module.add_function("ungetc", ft, None)
        });
        let c = b
            .build_call(fgetc_fn, &[fp.into()], "c")
            .unwrap()
            .try_as_basic_value()
            .basic()
            .unwrap()
            .into_int_value();
        // EOF (-1) or '\n' counts as eoln.
        let is_eof = b
            .build_int_compare(
                inkwell::IntPredicate::EQ,
                c,
                i32_ty.const_int((-1i32) as u64, true),
                "is_eof",
            )
            .unwrap();
        let is_nl = b
            .build_int_compare(
                inkwell::IntPredicate::EQ,
                c,
                i32_ty.const_int(10, false),
                "is_nl",
            )
            .unwrap();
        let result_chk = b.build_or(is_eof, is_nl, "or").unwrap();
        // Push it back if it wasn't EOF
        let push_bb = context.append_basic_block(func, "push");
        let chk_end_bb = context.append_basic_block(func, "chk_end");
        b.build_conditional_branch(is_eof, chk_end_bb, push_bb)
            .unwrap();
        b.position_at_end(push_bb);
        b.build_call(ungetc_fn, &[c.into(), fp.into()], "").unwrap();
        b.build_unconditional_branch(chk_end_bb).unwrap();
        b.position_at_end(chk_end_bb);
        b.build_unconditional_branch(end_bb).unwrap();
        b.position_at_end(end_bb);
        let phi = b.build_phi(i1_ty, "eoln_phi").unwrap();
        phi.add_incoming(&[
            (&i1_ty.const_int(1, false), null_bb),
            (&result_chk, chk_end_bb),
        ]);
        b.build_return(Some(&phi.as_basic_value())).unwrap();
    }

    // bruto_file_seek(f: ptr, pos: i64)
    {
        let fn_ty = void_ty.fn_type(&[ptr_ty.into(), i64_ty.into()], false);
        let func = module.add_function("bruto_file_seek", fn_ty, None);
        let bb = context.append_basic_block(func, "entry");
        let b = context.create_builder();
        b.position_at_end(bb);
        let s = func.get_nth_param(0).unwrap().into_pointer_value();
        let pos = func.get_nth_param(1).unwrap().into_int_value();
        let fp = b.build_load(ptr_ty, s, "fp").unwrap().into_pointer_value();
        let fseek_fn = module.get_function("fseek").unwrap_or_else(|| {
            let ft = i32_ty.fn_type(&[ptr_ty.into(), i64_ty.into(), i32_ty.into()], false);
            module.add_function("fseek", ft, None)
        });
        // SEEK_SET = 0
        b.build_call(
            fseek_fn,
            &[fp.into(), pos.into(), i32_ty.const_int(0, false).into()],
            "",
        )
        .unwrap();
        b.build_return(None).unwrap();
    }

    // bruto_file_filepos(f: ptr) -> i64
    {
        let fn_ty = i64_ty.fn_type(&[ptr_ty.into()], false);
        let func = module.add_function("bruto_file_filepos", fn_ty, None);
        let bb = context.append_basic_block(func, "entry");
        let b = context.create_builder();
        b.position_at_end(bb);
        let s = func.get_first_param().unwrap().into_pointer_value();
        let fp = b.build_load(ptr_ty, s, "fp").unwrap().into_pointer_value();
        let ftell_fn = module.get_function("ftell").unwrap_or_else(|| {
            let ft = i64_ty.fn_type(&[ptr_ty.into()], false);
            module.add_function("ftell", ft, None)
        });
        let r = b
            .build_call(ftell_fn, &[fp.into()], "r")
            .unwrap()
            .try_as_basic_value()
            .basic()
            .unwrap();
        b.build_return(Some(&r)).unwrap();
    }

    // bruto_file_filesize(f: ptr) -> i64
    {
        let fn_ty = i64_ty.fn_type(&[ptr_ty.into()], false);
        let func = module.add_function("bruto_file_filesize", fn_ty, None);
        let bb = context.append_basic_block(func, "entry");
        let b = context.create_builder();
        b.position_at_end(bb);
        let s = func.get_first_param().unwrap().into_pointer_value();
        let fp = b.build_load(ptr_ty, s, "fp").unwrap().into_pointer_value();
        let fseek_fn = module.get_function("fseek").unwrap();
        let ftell_fn = module.get_function("ftell").unwrap();
        // save pos, seek end, ftell, restore
        let saved = b
            .build_call(ftell_fn, &[fp.into()], "saved")
            .unwrap()
            .try_as_basic_value()
            .basic()
            .unwrap()
            .into_int_value();
        // SEEK_END = 2
        b.build_call(
            fseek_fn,
            &[
                fp.into(),
                i64_ty.const_int(0, false).into(),
                i32_ty.const_int(2, false).into(),
            ],
            "",
        )
        .unwrap();
        let size = b
            .build_call(ftell_fn, &[fp.into()], "size")
            .unwrap()
            .try_as_basic_value()
            .basic()
            .unwrap()
            .into_int_value();
        b.build_call(
            fseek_fn,
            &[fp.into(), saved.into(), i32_ty.const_int(0, false).into()],
            "",
        )
        .unwrap();
        b.build_return(Some(&size)).unwrap();
    }

    // bruto_range_check_fail(line: i64, lo: i64, hi: i64, val: i64) — abort with message
    {
        let fn_ty = void_ty.fn_type(
            &[i64_ty.into(), i64_ty.into(), i64_ty.into(), i64_ty.into()],
            false,
        );
        let func = module.add_function("bruto_range_check_fail", fn_ty, None);
        let bb = context.append_basic_block(func, "entry");
        let b = context.create_builder();
        b.position_at_end(bb);
        let line = func.get_nth_param(0).unwrap().into_int_value();
        let lo = func.get_nth_param(1).unwrap().into_int_value();
        let hi = func.get_nth_param(2).unwrap().into_int_value();
        let val = func.get_nth_param(3).unwrap().into_int_value();
        // fprintf(stderr, "Runtime error: range check failed at line %lld: %lld not in %lld..%lld\n", ...)
        let stderr_fp =
            crate::target::emit_load_stdio(&b, module, context, crate::target::Stdio::Stderr);
        let fprintf = module.get_function("fprintf").unwrap();
        let fmt = b
            .build_global_string_ptr(
                "Runtime error: range check failed at line %lld: %lld not in %lld..%lld\n",
                "rng_fmt",
            )
            .unwrap();
        b.build_call(
            fprintf,
            &[
                stderr_fp.into(),
                fmt.as_pointer_value().into(),
                line.into(),
                val.into(),
                lo.into(),
                hi.into(),
            ],
            "",
        )
        .unwrap();
        // exit(1)
        let exit_fn = module.get_function("exit").unwrap_or_else(|| {
            let ft = void_ty.fn_type(&[i32_ty.into()], false);
            module.add_function("exit", ft, None)
        });
        b.build_call(exit_fn, &[i32_ty.const_int(1, false).into()], "")
            .unwrap();
        b.build_unreachable().unwrap();
    }

    // bruto_overflow_fail(line: i64) — abort
    {
        let fn_ty = void_ty.fn_type(&[i64_ty.into()], false);
        let func = module.add_function("bruto_overflow_fail", fn_ty, None);
        let bb = context.append_basic_block(func, "entry");
        let b = context.create_builder();
        b.position_at_end(bb);
        let line = func.get_first_param().unwrap().into_int_value();
        let stderr_fp =
            crate::target::emit_load_stdio(&b, module, context, crate::target::Stdio::Stderr);
        let fprintf = module.get_function("fprintf").unwrap();
        let fmt = b
            .build_global_string_ptr("Runtime error: integer overflow at line %lld\n", "ovf_fmt")
            .unwrap();
        b.build_call(
            fprintf,
            &[stderr_fp.into(), fmt.as_pointer_value().into(), line.into()],
            "",
        )
        .unwrap();
        let exit_fn = module.get_function("exit").unwrap_or_else(|| {
            let ft = void_ty.fn_type(&[i32_ty.into()], false);
            module.add_function("exit", ft, None)
        });
        b.build_call(exit_fn, &[i32_ty.const_int(1, false).into()], "")
            .unwrap();
        b.build_unreachable().unwrap();
    }

    // bruto_file_read_str(f: ptr) -> ptr  (reads a line)
    {
        let fn_ty = ptr_ty.fn_type(&[ptr_ty.into()], false);
        let func = module.add_function("bruto_file_read_str", fn_ty, None);
        let bb = context.append_basic_block(func, "entry");
        let b = context.create_builder();
        b.position_at_end(bb);
        let s = func.get_first_param().unwrap().into_pointer_value();
        let fp = b.build_load(ptr_ty, s, "fp").unwrap().into_pointer_value();
        let malloc = module.get_function("malloc").unwrap();
        let buf = b
            .build_call(malloc, &[i64_ty.const_int(1024, false).into()], "buf")
            .unwrap()
            .try_as_basic_value()
            .basic()
            .unwrap()
            .into_pointer_value();
        b.build_store(buf, context.i8_type().const_int(0, false))
            .unwrap();
        let fgets_fn = module.get_function("fgets").unwrap_or_else(|| {
            let ft = ptr_ty.fn_type(&[ptr_ty.into(), i32_ty.into(), ptr_ty.into()], false);
            module.add_function("fgets", ft, None)
        });
        b.build_call(
            fgets_fn,
            &[buf.into(), i32_ty.const_int(1024, false).into(), fp.into()],
            "",
        )
        .unwrap();
        let strlen = module.get_function("strlen").unwrap();
        let len = b
            .build_call(strlen, &[buf.into()], "L")
            .unwrap()
            .try_as_basic_value()
            .basic()
            .unwrap()
            .into_int_value();
        let zero_check = b
            .build_int_compare(
                inkwell::IntPredicate::EQ,
                len,
                i64_ty.const_int(0, false),
                "is0",
            )
            .unwrap();
        let not_zero_bb = context.append_basic_block(func, "not_zero");
        let done_bb = context.append_basic_block(func, "done");
        b.build_conditional_branch(zero_check, done_bb, not_zero_bb)
            .unwrap();
        b.position_at_end(not_zero_bb);
        let last_idx = b
            .build_int_sub(len, i64_ty.const_int(1, false), "li")
            .unwrap();
        let last_ptr = unsafe {
            b.build_in_bounds_gep(context.i8_type(), buf, &[last_idx], "lp")
                .unwrap()
        };
        let last_ch = b
            .build_load(context.i8_type(), last_ptr, "lc")
            .unwrap()
            .into_int_value();
        let is_nl = b
            .build_int_compare(
                inkwell::IntPredicate::EQ,
                last_ch,
                context.i8_type().const_int(10, false),
                "isnl",
            )
            .unwrap();
        let strip_bb = context.append_basic_block(func, "strip");
        b.build_conditional_branch(is_nl, strip_bb, done_bb)
            .unwrap();
        b.position_at_end(strip_bb);
        b.build_store(last_ptr, context.i8_type().const_int(0, false))
            .unwrap();
        b.build_unconditional_branch(done_bb).unwrap();
        b.position_at_end(done_bb);
        b.build_return(Some(&buf)).unwrap();
    }
}

/// Helper: emit a `bruto_capture_write_*` function that fprintf's a single
/// value through a format string to the capture file (if open).
fn emit_capture_write_fn<'ctx>(
    context: &'ctx Context,
    module: &Module<'ctx>,
    name: &str,
    param_types: &[inkwell::types::BasicMetadataTypeEnum<'ctx>],
    fmt_str: &str,
) {
    let void_ty = context.void_type();
    let ptr_ty = context.ptr_type(AddressSpace::default());
    let fn_ty = void_ty.fn_type(param_types, false);
    let func = module.add_function(name, fn_ty, None);
    let bb = context.append_basic_block(func, "entry");
    let b = context.create_builder();
    b.position_at_end(bb);

    let g = module.get_global("_bruto_capture_fp").unwrap();
    let fp = b
        .build_load(ptr_ty, g.as_pointer_value(), "fp")
        .unwrap()
        .into_pointer_value();
    let is_null = b.build_is_null(fp, "null").unwrap();

    let write_bb = context.append_basic_block(func, "write");
    let end_bb = context.append_basic_block(func, "end");
    b.build_conditional_branch(is_null, end_bb, write_bb)
        .unwrap();

    b.position_at_end(write_bb);
    let fprintf = module.get_function("fprintf").unwrap();
    let fmt = b.build_global_string_ptr(fmt_str, "cap_fmt").unwrap();
    let val = func.get_first_param().unwrap();
    b.build_call(
        fprintf,
        &[fp.into(), fmt.as_pointer_value().into(), val.into()],
        "",
    )
    .unwrap();
    let fflush = module.get_function("fflush").unwrap();
    b.build_call(fflush, &[fp.into()], "").unwrap();
    b.build_unconditional_branch(end_bb).unwrap();

    b.position_at_end(end_bb);
    b.build_return(None).unwrap();
}
