(function() {var implementors = {};
implementors["hdf5"] = [{"text":"impl&lt;'_&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;&amp;'_ <a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.str.html\">str</a>&gt; for <a class=\"enum\" href=\"hdf5/enum.Error.html\" title=\"enum hdf5::Error\">Error</a>","synthetic":false,"types":["hdf5::error::Error"]},{"text":"impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"struct\" href=\"https://doc.rust-lang.org/nightly/alloc/string/struct.String.html\" title=\"struct alloc::string::String\">String</a>&gt; for <a class=\"enum\" href=\"hdf5/enum.Error.html\" title=\"enum hdf5::Error\">Error</a>","synthetic":false,"types":["hdf5::error::Error"]},{"text":"impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"struct\" href=\"https://docs.rs/ndarray/0.13/ndarray/error/struct.ShapeError.html\" title=\"struct ndarray::error::ShapeError\">ShapeError</a>&gt; for <a class=\"enum\" href=\"hdf5/enum.Error.html\" title=\"enum hdf5::Error\">Error</a>","synthetic":false,"types":["hdf5::error::Error"]},{"text":"impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;H5T_order_t&gt; for <a class=\"enum\" href=\"hdf5/datatype/enum.ByteOrder.html\" title=\"enum hdf5::datatype::ByteOrder\">ByteOrder</a>","synthetic":false,"types":["hdf5::hl::datatype::ByteOrder"]},{"text":"impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;H5D_vds_view_t&gt; for <a class=\"enum\" href=\"hdf5/dataset/enum.VirtualView.html\" title=\"enum hdf5::dataset::VirtualView\">VirtualView</a>","synthetic":false,"types":["hdf5::hl::plist::dataset_access::VirtualView"]},{"text":"impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"enum\" href=\"hdf5/dataset/enum.VirtualView.html\" title=\"enum hdf5::dataset::VirtualView\">VirtualView</a>&gt; for H5D_vds_view_t","synthetic":false,"types":["hdf5_sys::h5d::hdf5_1_10_0::H5D_vds_view_t"]},{"text":"impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;H5F_close_degree_t&gt; for <a class=\"enum\" href=\"hdf5/file/enum.FileCloseDegree.html\" title=\"enum hdf5::file::FileCloseDegree\">FileCloseDegree</a>","synthetic":false,"types":["hdf5::hl::plist::file_access::FileCloseDegree"]},{"text":"impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;H5C_cache_incr_mode&gt; for <a class=\"enum\" href=\"hdf5/file/enum.CacheIncreaseMode.html\" title=\"enum hdf5::file::CacheIncreaseMode\">CacheIncreaseMode</a>","synthetic":false,"types":["hdf5::hl::plist::file_access::CacheIncreaseMode"]},{"text":"impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;H5C_cache_flash_incr_mode&gt; for <a class=\"enum\" href=\"hdf5/file/enum.FlashIncreaseMode.html\" title=\"enum hdf5::file::FlashIncreaseMode\">FlashIncreaseMode</a>","synthetic":false,"types":["hdf5::hl::plist::file_access::FlashIncreaseMode"]},{"text":"impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;H5C_cache_decr_mode&gt; for <a class=\"enum\" href=\"hdf5/file/enum.CacheDecreaseMode.html\" title=\"enum hdf5::file::CacheDecreaseMode\">CacheDecreaseMode</a>","synthetic":false,"types":["hdf5::hl::plist::file_access::CacheDecreaseMode"]},{"text":"impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.i32.html\">i32</a>&gt; for <a class=\"enum\" href=\"hdf5/file/enum.MetadataWriteStrategy.html\" title=\"enum hdf5::file::MetadataWriteStrategy\">MetadataWriteStrategy</a>","synthetic":false,"types":["hdf5::hl::plist::file_access::MetadataWriteStrategy"]},{"text":"impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;H5AC_cache_config_t&gt; for <a class=\"struct\" href=\"hdf5/file/struct.MetadataCacheConfig.html\" title=\"struct hdf5::file::MetadataCacheConfig\">MetadataCacheConfig</a>","synthetic":false,"types":["hdf5::hl::plist::file_access::MetadataCacheConfig"]},{"text":"impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;H5AC_cache_image_config_t&gt; for <a class=\"struct\" href=\"hdf5/file/struct.CacheImageConfig.html\" title=\"struct hdf5::file::CacheImageConfig\">CacheImageConfig</a>","synthetic":false,"types":["hdf5::hl::plist::file_access::cache_image_config::CacheImageConfig"]},{"text":"impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;H5F_libver_t&gt; for <a class=\"enum\" href=\"hdf5/file/enum.LibraryVersion.html\" title=\"enum hdf5::file::LibraryVersion\">LibraryVersion</a>","synthetic":false,"types":["hdf5::hl::plist::file_access::libver::LibraryVersion"]}];

            if (window.register_implementors) {
                window.register_implementors(implementors);
            } else {
                window.pending_implementors = implementors;
            }
        })()